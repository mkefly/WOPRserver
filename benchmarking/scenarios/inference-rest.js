import { readTestData } from "../common/helpers.js";
import { RestClient } from "../common/rest.js";
import { Trend, Counter, Gauge } from "k6/metrics";

const TestData = {
  iris: readTestData("iris"),
};

const rest = new RestClient();

const ScenarioDuration = "60s";
const ScenarioVUs = 300;

export const options = {
  scenarios: {
    iris_rest: {
      executor: "constant-vus",
      duration: ScenarioDuration,
      vus: ScenarioVUs,
      tags: { model_name: "iris", protocol: "rest" },
      env: { MODEL_NAME: "iris", PROTOCOL: "rest" },
    },
  },
  thresholds: {
    http_reqs: ["rate > 1100"],
  },
};

// Worker-aware unary metrics
const unaryWorkerReqs = new Counter("unary_worker_reqs");
const unaryWorkerProcRssKb = new Trend("unary_worker_proc_rss_kb");
const unaryWorkerProcCpuPct = new Trend("unary_worker_proc_cpu_pct");
const unaryWorkersSeenLocal = new Trend("unary_workers_seen_local");
const unaryDbgWorkersSeen = new Gauge("unary_debug_workers_seen_last_batch");

export function setup() {
  rest.loadModel("iris");
  return TestData;
}

export default function (data) {
  const modelName = __ENV.MODEL_NAME;

  const res = rest.infer(modelName, data[modelName].rest);

  const seenWorkers = new Set();

  try {
    const body = JSON.parse(res.body);
    const outputs = body.outputs || [];
    const params = outputs[0]?.parameters || {};
    const headers = params.headers || {};
    const pid = headers["X-Worker-PID"] || "unknown";
    seenWorkers.add(pid);
    unaryWorkerReqs.add(1, { worker: pid, model_name: modelName });

    const rss = Number(headers["X-Proc-RSS-KB"]);
    if (!Number.isNaN(rss)) {
      unaryWorkerProcRssKb.add(rss, { worker: pid, model_name: modelName });
    }
    const cpu = Number(headers["X-Proc-CPU-Pct"]);
    if (!Number.isNaN(cpu)) {
      unaryWorkerProcCpuPct.add(cpu, { worker: pid, model_name: modelName });
    }
  } catch (_) {
    // ignore parse issues
  }

  unaryWorkersSeenLocal.add(seenWorkers.size, { model_name: modelName });
  unaryDbgWorkersSeen.add(seenWorkers.size, { model_name: modelName });
}

export function teardown(data) {
  rest.unloadModel("iris");
}

/* ------------------------- Summary (REST unary) ------------------------- */

export function handleSummary(data) {

  const lines = [];
  lines.push("=== REST Unary Summary ===");
  lines.push(`Approx. distinct workers (max per-VU batch): ${approxWorkers}`);
  if (avgRss !== null) lines.push(`Average worker RSS (KB): ${avgRss.toFixed(0)}`);
  if (avgCpu !== null) lines.push(`Average worker CPU (%): ${avgCpu.toFixed(2)}`);
  lines.push("");

  // Classic k6 table
  const classic = textSummary(data, { indent: " ", enableColors: true });

  // Single stdout output containing both sections
  return { stdout: `${lines.join("\n")}\n${classic}` };
}
