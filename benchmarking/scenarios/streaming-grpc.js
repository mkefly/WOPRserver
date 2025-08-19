import { sleep } from "k6";
import http from "k6/http";
import { readTestData } from "../common/helpers.js";
import { GrpcClient } from "../common/grpc.js";
import { Trend, Counter, Gauge } from "k6/metrics";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

/**
 * gRPC streaming for BOTH models:
 *   - summodel  (numeric ramp streaming)
 *   - llm       (ToyLLM text streaming)
 *
 * Adds per-worker metrics (from model headers) AND a 1Hz Prometheus poller.
 */

const TestData = {
  summodel: readTestData("summodel"),
  llm: readTestData("llm"),
};

const grpc = new GrpcClient();

// -------- Global stream metrics
const streamMsgs = new Counter("grpc_stream_msgs");
const ttft = new Trend("grpc_stream_first_msg_ms");

// -------- Per-worker (from response headers)
const grpcWorkerMsgs = new Counter("grpc_worker_msgs");
const grpcWorkerFirstMsgMs = new Trend("grpc_worker_first_msg_ms");
const grpcWorkerStreamDurationMs = new Trend("grpc_worker_stream_duration_ms");
const grpcWorkerStreams = new Counter("grpc_worker_streams");
const grpcWorkerProcRssKb = new Trend("grpc_worker_proc_rss_kb");
const grpcWorkerProcCpuPct = new Trend("grpc_worker_proc_cpu_pct");

// -------- Distinct workers this VU saw per stream
const grpcWorkersSeenLocal = new Trend("grpc_workers_seen_local");
const dbgWorkersSeen = new Gauge("grpc_debug_workers_seen_last_stream");

// -------- Prometheus-scraped worker/pool/request stats (1 Hz)
const promWorkersActive = new Trend("grpc_workers_active_prom");     // #workers with active streams
const promWorkersDistinct = new Trend("grpc_workers_distinct_prom"); // distinct PIDs
const promPoolsActive = new Trend("grpc_pools_active_prom");
const promModelsLoaded = new Trend("grpc_models_loaded_total_prom");
const promModelsUnloaded = new Trend("grpc_models_unloaded_total_prom");
const promStreamReqs200 = new Trend("grpc_stream_reqs_200_prom");    // REST stream totals still useful (server-wide)
const promStreamReqsOther = new Trend("grpc_stream_reqs_other_prom");

export const options = {
  scenarios: {
    sum_grpc_stream: {
      executor: "constant-vus",
      duration: "60s",
      vus: 40,
      tags: { model_name: "summodel", protocol: "grpc-stream" },
      env: { MODEL_NAME: "summodel" },
    },
    llm_grpc_stream: {
      executor: "constant-vus",
      duration: "60s",
      vus: 20,
      tags: { model_name: "llm", protocol: "grpc-stream" },
      env: { MODEL_NAME: "llm" },
    },
    // Lightweight /metrics poller; ~1 request/sec
    prom_scrape: {
      executor: "constant-vus",
      duration: "60s",
      vus: 1,
      exec: "promScraper",
      tags: { role: "prom" },
    },
  },
  thresholds: {
    grpc_stream_msgs: ["count > 300"],
  },
};

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function setup() {
  grpc.loadModel("summodel");
  grpc.loadModel("llm");
  return TestData;
}

export default function (data) {
  const modelName = __ENV.MODEL_NAME;
  const d = data[modelName];

  const payload =
    d.grpc_many && d.grpc_many.length ? pickRandom(d.grpc_many) : d.grpc;

  const start = Date.now();
  let seenFirst = false;
  const firstSeenByWorker = {};
  const seenWorkers = new Set();

  grpc.streamInfer(payload, {
    onData: (message) => {
      streamMsgs.add(1);

      if (!seenFirst) {
        ttft.add(Date.now() - start, { model_name: modelName });
        seenFirst = true;
      }

      try {
        const outputs = message.outputs || [];
        const params = outputs[0] && outputs[0].parameters ? outputs[0].parameters : {};
        const headers = params.headers || {};
        const pid = headers["X-Worker-PID"] || "unknown";

        seenWorkers.add(pid);
        grpcWorkerMsgs.add(1, { worker: pid, model_name: modelName });

        if (!firstSeenByWorker[pid]) {
          firstSeenByWorker[pid] = Date.now();
          grpcWorkerFirstMsgMs.add(firstSeenByWorker[pid] - start, {
            worker: pid,
            model_name: modelName,
          });
        }

        // Process telemetry
        const rss = Number(headers["X-Proc-RSS-KB"]);
        if (!Number.isNaN(rss)) {
          grpcWorkerProcRssKb.add(rss, { worker: pid, model_name: modelName });
        }
        const cpu = Number(headers["X-Proc-CPU-Pct"]);
        if (!Number.isNaN(cpu)) {
          grpcWorkerProcCpuPct.add(cpu, { worker: pid, model_name: modelName });
        }
      } catch (_) {
        // ignore parse issues
      }
    },
    onError: (e) => {
      throw e;
    },
  });

  const dur = Date.now() - start;
  seenWorkers.forEach((pid) => {
    grpcWorkerStreamDurationMs.add(dur, { worker: pid, model_name: modelName });
    grpcWorkerStreams.add(1, { worker: pid, model_name: modelName });
  });

  grpcWorkersSeenLocal.add(seenWorkers.size, { model_name: modelName });
  dbgWorkersSeen.add(seenWorkers.size, { model_name: modelName });

  sleep(0.1);
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

  // Workers
  const active = countActiveWorkersFromProm(body);
  const distinct = countDistinctWorkersFromProm(body);
  if (active !== null) promWorkersActive.add(active);
  if (distinct !== null) promWorkersDistinct.add(distinct);

  // Pools
  const pools = sumInferencePoolsFromProm(body);
  if (pools !== null) promPoolsActive.add(pools);

  // Model update counters
  const loads = parsePromMetric(body, 'worker_model_updates_total{outcome="success",type="Load"}');
  const unloads = parsePromMetric(body, 'worker_model_updates_total{outcome="success",type="Unload"}');
  if (loads !== null) promModelsLoaded.add(loads);
  if (unloads !== null) promModelsUnloaded.add(unloads);

  // REST streaming totals (server-wide context)
  const reqStats = parseRestStreamRequests(body);
  if (reqStats) {
    if (typeof reqStats.ok200 === "number") promStreamReqs200.add(reqStats.ok200);
    if (typeof reqStats.other === "number") promStreamReqsOther.add(reqStats.other);
  }

  sleep(1);
}

/* ------------------------- Summary & helpers ------------------------- */

function parsePromMetric(body, exactLinePrefix) {
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
    (data.metrics["grpc_workers_seen_local"] && data.metrics["grpc_workers_seen_local"].values && data.metrics["grpc_workers_seen_local"].values.max) || 0;
  const avgRss =
    (data.metrics["grpc_worker_proc_rss_kb"] && data.metrics["grpc_worker_proc_rss_kb"].values && data.metrics["grpc_worker_proc_rss_kb"].values.avg) || null;
  const avgCpu =
    (data.metrics["grpc_worker_proc_cpu_pct"] && data.metrics["grpc_worker_proc_cpu_pct"].values && data.metrics["grpc_worker_proc_cpu_pct"].values.avg) || null;

  // Prom poller aggregates
  const activeAvg = data.metrics["grpc_workers_active_prom"]?.values?.avg ?? null;
  const activeMax = data.metrics["grpc_workers_active_prom"]?.values?.max ?? null;
  const distinctMax = data.metrics["grpc_workers_distinct_prom"]?.values?.max ?? null;
  const poolsAvg = data.metrics["grpc_pools_active_prom"]?.values?.avg ?? null;
  const poolsMax = data.metrics["grpc_pools_active_prom"]?.values?.max ?? null;
  const loadsMax = data.metrics["grpc_models_loaded_total_prom"]?.values?.max ?? null;
  const unloadsMax = data.metrics["grpc_models_unloaded_total_prom"]?.values?.max ?? null;
  const req200Max = data.metrics["grpc_stream_reqs_200_prom"]?.values?.max ?? null;
  const reqOtherMax = data.metrics["grpc_stream_reqs_other_prom"]?.values?.max ?? null;

  const lines = [];
  lines.push("=== Streaming gRPC Summary ===");
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

  return {
    stdout: `${lines.join("\n")}\n${textSummary(data, { indent: " ", enableColors: true })}`,
  };
}

export function teardown() {
  grpc.unloadModel("summodel");
  grpc.unloadModel("llm");
  grpc.close();
}
