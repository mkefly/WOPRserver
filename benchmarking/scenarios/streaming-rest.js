import { sleep } from "k6";
import http from "k6/http";
import { readTestData } from "../common/helpers.js";
import { RestClient } from "../common/rest.js";
import { Trend, Counter, Gauge } from "k6/metrics";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

/**
 * REST SSE streaming for BOTH models:
 *   - summodel (numeric ramp)
 *   - llm      (ToyLLM text)
 *
 * Adds per-worker metrics (from model headers) AND a 1Hz Prometheus poller.
 */

const TestData = {
  summodel: readTestData("summodel"),
  llm: readTestData("llm"),
};

const rest = new RestClient();

// ---------- Global (overall) metrics
const sseEvents = new Counter("sse_events");
const sseFirstEventMs = new Trend("sse_first_event_ms");
const sseTotalDurationMs = new Trend("sse_total_duration_ms");

// ---------- Per-worker (tagged by worker + model)
const workerEvents = new Counter("worker_events");
const workerFirstEventMs = new Trend("worker_first_event_ms");
const workerChunkSizeBytes = new Trend("worker_chunk_size_bytes");
const workerBytes = new Counter("worker_bytes");
const workerStreamDurationMs = new Trend("worker_stream_duration_ms");
const workerStreams = new Counter("worker_streams");

// ---------- Per-worker process telemetry (from model headers)
const workerProcRssKb = new Trend("worker_proc_rss_kb");
const workerProcCpuPct = new Trend("worker_proc_cpu_pct");

// ---------- Distinct workers this VU saw per stream
const workersSeenLocal = new Trend("workers_seen_local");
const dbgWorkersSeen = new Gauge("debug_workers_seen_last_stream");

// ---------- Prometheus-scraped stats (1 Hz)
const promWorkersActive = new Trend("rest_workers_active_prom");     // #workers with active streams
const promWorkersDistinct = new Trend("rest_workers_distinct_prom"); // distinct PIDs seen in Prom
const promPoolsActive = new Trend("rest_pools_active_prom");         // sum of inference_pools_active{pid=...}
const promModelsLoaded = new Trend("rest_models_loaded_total_prom"); // samples of counter
const promModelsUnloaded = new Trend("rest_models_unloaded_total_prom");
const promStreamReqs200 = new Trend("rest_stream_reqs_200_prom");   // samples of counter
const promStreamReqsOther = new Trend("rest_stream_reqs_other_prom");

// ---------- K6 scenarios
export const options = {
  scenarios: {
    sum_rest_stream: {
      executor: "constant-vus",
      duration: "60s",
      vus: 40,
      tags: { model_name: "summodel", protocol: "rest-stream" },
      env: { MODEL_NAME: "summodel" },
    },
    llm_rest_stream: {
      executor: "constant-vus",
      duration: "60s",
      vus: 20,
      tags: { model_name: "llm", protocol: "rest-stream" },
      env: { MODEL_NAME: "llm" },
    },
    // Lightweight /metrics poller; ~1 request/sec for the whole test
    prom_scrape: {
      executor: "constant-vus",
      duration: "60s",
      vus: 1,
      exec: "promScraper",
      tags: { role: "prom" },
    },
  },
  thresholds: {
    sse_events: ["count > 300"], // sanity
  },
};

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function setup() {
  rest.loadModel("summodel");
  rest.loadModel("llm");
  return TestData;
}

export default function (data) {
  const modelName = __ENV.MODEL_NAME;
  const d = data[modelName];

  const payload =
    d.rest_many && d.rest_many.length ? pickRandom(d.rest_many) : d.rest;

  streamOnce(modelName, payload);
  sleep(0.1);
}

function streamOnce(modelName, payload) {
  const t0 = Date.now();
  let firstSeenGlobal = null;
  const firstSeenByWorker = {}; // pid -> ts
  const seenWorkers = new Set();

  rest.inferStream(
    modelName,
    payload,
    (event) => {
      sseEvents.add(1);
      if (firstSeenGlobal === null) {
        firstSeenGlobal = Date.now();
        sseFirstEventMs.add(firstSeenGlobal - t0, { model_name: modelName });
      }

      // Parse the SSE "data:" JSON to read headers injected by the model.
      let pid = "unknown";
      try {
        const msg = JSON.parse(event.data);
        const outputs = msg.outputs || [];
        const params = outputs[0] && outputs[0].parameters ? outputs[0].parameters : {};
        const headers = params.headers || {};
        pid = headers["X-Worker-PID"] || "unknown";

        // Per-worker bookkeeping
        seenWorkers.add(pid);

        // Chunk size & bytes per worker (approx via JSON string length)
        const sz = event.data.length;
        workerChunkSizeBytes.add(sz, { worker: pid, model_name: modelName });
        workerBytes.add(sz, { worker: pid, model_name: modelName });

        // First-chunk latency per worker
        if (!firstSeenByWorker[pid]) {
          firstSeenByWorker[pid] = Date.now();
          workerFirstEventMs.add(firstSeenByWorker[pid] - t0, {
            worker: pid,
            model_name: modelName,
          });
        }

        workerEvents.add(1, { worker: pid, model_name: modelName });

        // Process telemetry
        const rss = Number(headers["X-Proc-RSS-KB"]);
        if (!Number.isNaN(rss)) {
          workerProcRssKb.add(rss, { worker: pid, model_name: modelName });
        }
        const cpu = Number(headers["X-Proc-CPU-Pct"]);
        if (!Number.isNaN(cpu)) {
          workerProcCpuPct.add(cpu, { worker: pid, model_name: modelName });
        }
      } catch (_) {
        // If parsing fails, still count global metrics.
      }
    },
    (err) => {
      throw err;
    }
  );

  const dur = Date.now() - t0;
  sseTotalDurationMs.add(dur, { model_name: modelName });

  // Attribute stream duration / streams to each contributing worker
  seenWorkers.forEach((pid) => {
    workerStreamDurationMs.add(dur, { worker: pid, model_name: modelName });
    workerStreams.add(1, { worker: pid, model_name: modelName });
  });

  // Distinct workers in this stream (VU-local)
  workersSeenLocal.add(seenWorkers.size, { model_name: modelName });
  dbgWorkersSeen.add(seenWorkers.size, { model_name: modelName });
}

/* ------------------------- Prometheus poller ------------------------- */

export function promScraper() {
  const METRICS_PORT = __ENV.MLSERVER_METRICS_PORT || __ENV.MLSERVER_HTTP_PORT;
  const base = `http://${__ENV.MLSERVER_HOST}:${METRICS_PORT}`;
  const res = http.get(`${base}/metrics`, { timeout: "2s" });
  if (!res || res.status !== 200) {
    sleep(1);
    return;
  }
  const body = res.body;

  // Workers (from worker_active_streams)
  const active = countActiveWorkersFromProm(body);
  const distinct = countDistinctWorkersFromProm(body);
  if (active !== null) promWorkersActive.add(active);
  if (distinct !== null) promWorkersDistinct.add(distinct);

  // Inference pools (sum over pids)
  const pools = sumInferencePoolsFromProm(body);
  if (pools !== null) promPoolsActive.add(pools);

  // Model (load/unload) counters
  const loads = parsePromMetric(body, 'worker_model_updates_total{outcome="success",type="Load"}');
  const unloads = parsePromMetric(body, 'worker_model_updates_total{outcome="success",type="Unload"}');
  if (loads !== null) promModelsLoaded.add(loads);
  if (unloads !== null) promModelsUnloaded.add(unloads);

  // Streaming REST request totals (by status) for /v2/models/*/infer_stream
  const reqStats = parseRestStreamRequests(body);
  if (reqStats) {
    if (typeof reqStats.ok200 === "number") promStreamReqs200.add(reqStats.ok200);
    if (typeof reqStats.other === "number") promStreamReqsOther.add(reqStats.other);
  }

  sleep(1); // ~1Hz
}

/* ------------------------- Summary & helpers ------------------------- */

function parsePromMetric(body, exactLinePrefix) {
  // exactLinePrefix can be a full metric with labels (e.g., worker_model_updates_total{...})
  const re = new RegExp(`^${escapeRe(exactLinePrefix)}\\s+([0-9eE+\\-.]+)\\s*$`, "m");
  const m = body && body.match(re);
  return m ? Number(m[1]) : null;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function countDistinctWorkersFromProm(body) {
  if (!body) return null;
  const re = /^worker_active_streams\{[^}]*pid="([^"]+)"[^}]*\}\s+[0-9eE+\-.]+\s*$/gm;
  const set = new Set();
  let m;
  while ((m = re.exec(body)) !== null) set.add(m[1]);
  return set.size || null;
}

function countActiveWorkersFromProm(body) {
  if (!body) return null;
  const re = /^worker_active_streams\{[^}]*pid="([^"]+)"[^}]*\}\s+([0-9eE+\-.]+)\s*$/gm;
  let active = 0;
  let m;
  while ((m = re.exec(body)) !== null) {
    const v = Number(m[2]);
    if (!Number.isNaN(v) && v > 0) active += 1;
  }
  return active;
}

function sumInferencePoolsFromProm(body) {
  if (!body) return null;
  const re = /^inference_pools_active\{[^}]*pid="[^"]+"[^}]*\}\s+([0-9eE+\-.]+)\s*$/gm;
  let sum = 0;
  let saw = false;
  let m;
  while ((m = re.exec(body)) !== null) {
    const v = Number(m[1]);
    if (!Number.isNaN(v)) {
      sum += v;
      saw = true;
    }
  }
  return saw ? sum : null;
}

function parseLabels(labelStr) {
  // Parse key="value" pairs; ignores escaped quotes inside values.
  const map = {};
  const re = /(\w+)="([^"]*)"/g;
  let m;
  while ((m = re.exec(labelStr)) !== null) map[m[1]] = m[2];
  return map;
}

function parseRestStreamRequests(body) {
  if (!body) return null;
  const re = /^rest_server_requests_total\{([^}]*)\}\s+([0-9eE+\-.]+)\s*$/gm;
  let ok200 = 0;
  let other = 0;
  let saw = false;
  let m;
  while ((m = re.exec(body)) !== null) {
    const labels = parseLabels(m[1]);
    const val = Number(m[2]);
    if (Number.isNaN(val)) continue;
    // Focus on /v2/models/{name}/infer_stream
    const path = labels.path || "";
    if (path.indexOf("/v2/models/") === 0 && path.indexOf("/infer_stream") !== -1) {
      const sc = labels.status_code || "";
      if (sc === "200") ok200 += val;
      else other += val;
      saw = true;
    }
  }
  return saw ? { ok200, other, total: ok200 + other } : null;
}

export function handleSummary(data) {
  const METRICS_PORT = __ENV.MLSERVER_METRICS_PORT || __ENV.MLSERVER_HTTP_PORT;
  const base = `http://${__ENV.MLSERVER_HOST}:${METRICS_PORT}`;
  let prom = null;
  try {
    const res = http.get(`${base}/metrics`, { timeout: "3s" });
    prom = res && res.status === 200 ? res.body : null;
  } catch (_) {
    prom = null;
  }

  const approxWorkers =
    (data.metrics["workers_seen_local"] && data.metrics["workers_seen_local"].values && data.metrics["workers_seen_local"].values.max) || 0;
  const avgRss =
    (data.metrics["worker_proc_rss_kb"] && data.metrics["worker_proc_rss_kb"].values && data.metrics["worker_proc_rss_kb"].values.avg) || null;
  const avgCpu =
    (data.metrics["worker_proc_cpu_pct"] && data.metrics["worker_proc_cpu_pct"].values && data.metrics["worker_proc_cpu_pct"].values.avg) || null;

  // Prom poller aggregates
  const activeAvg = data.metrics["rest_workers_active_prom"]?.values?.avg ?? null;
  const activeMax = data.metrics["rest_workers_active_prom"]?.values?.max ?? null;
  const distinctMax = data.metrics["rest_workers_distinct_prom"]?.values?.max ?? null;
  const poolsAvg = data.metrics["rest_pools_active_prom"]?.values?.avg ?? null;
  const poolsMax = data.metrics["rest_pools_active_prom"]?.values?.max ?? null;
  const loadsMax = data.metrics["rest_models_loaded_total_prom"]?.values?.max ?? null;
  const unloadsMax = data.metrics["rest_models_unloaded_total_prom"]?.values?.max ?? null;
  const req200Max = data.metrics["rest_stream_reqs_200_prom"]?.values?.max ?? null;
  const reqOtherMax = data.metrics["rest_stream_reqs_other_prom"]?.values?.max ?? null;

  const lines = [];
  lines.push("=== Streaming REST Summary ===");
  lines.push(`VU-local distinct workers (max per stream): ${approxWorkers}`);
  if (avgRss !== null) lines.push(`Avg worker RSS (KB): ${avgRss.toFixed(0)}`);
  if (avgCpu !== null) lines.push(`Avg worker CPU (%): ${avgCpu.toFixed(2)}`);

  lines.push("");
  lines.push("Prometheus (polled during test):");
  if (activeAvg !== null || activeMax !== null) {
    lines.push(`  Workers active (avg/max): ${activeAvg?.toFixed?.(2) ?? "n/a"} / ${activeMax ?? "n/a"}`);
  }
  if (distinctMax !== null) lines.push(`  Workers discovered (max distinct PIDs): ${distinctMax}`);
  if (poolsAvg !== null || poolsMax !== null) {
    lines.push(`  Inference pools active (avg/max): ${poolsAvg?.toFixed?.(2) ?? "n/a"} / ${poolsMax ?? "n/a"}`);
  }
  if (loadsMax !== null || unloadsMax !== null) {
    lines.push(`  Model updates: loads=${loadsMax ?? "n/a"}, unloads=${unloadsMax ?? "n/a"}`);
  }
  if (req200Max !== null || reqOtherMax !== null) {
    lines.push(`  REST /infer_stream requests: 200=${req200Max ?? 0}, non-200=${reqOtherMax ?? 0}, total=${(req200Max ?? 0) + (reqOtherMax ?? 0)}`);
  }

  lines.push("");
  if (prom) {
    lines.push("Snapshot (/metrics at end):");
    const poolsSnap = sumInferencePoolsFromProm(prom);
    const reqSnap = parseRestStreamRequests(prom);
    if (poolsSnap !== null) lines.push(`  Inference pools active (snapshot sum): ${poolsSnap}`);
    if (reqSnap) {
      lines.push(`  REST /infer_stream requests (snapshot): 200=${reqSnap.ok200}, non-200=${reqSnap.other}, total=${reqSnap.total}`);
    }
  } else {
    lines.push("Prometheus scrape not available (no /metrics or timed out).");
  }
  lines.push("");

  // Classic k6 table + our summary to STDOUT only
  return {
    stdout: `${lines.join("\n")}\n${textSummary(data, { indent: " ", enableColors: true })}`,
  };
}

export function teardown() {
  rest.unloadModel("summodel");
  rest.unloadModel("llm");
}
