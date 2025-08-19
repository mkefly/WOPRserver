import { readTestData } from "../common/helpers.js";
import { GrpcClient } from "../common/grpc.js";
import { Counter, Trend, Gauge } from "k6/metrics";

const TestData = {
  iris: readTestData("iris"),
};

const grpc = new GrpcClient();

const ScenarioDuration = "60s";
const ScenarioVUs = 300;

export const options = {
  scenarios: {
    iris_grpc: {
      executor: "constant-vus",
      duration: ScenarioDuration,
      vus: ScenarioVUs,
      tags: { model_name: "iris", protocol: "grpc" },
      env: { MODEL_NAME: "iris", PROTOCOL: "grpc" },
    },
  },
  thresholds: {
    // Adjust so that it fits within the resources available in GH Actions
    grpc_reqs: ["rate > 1500"],
  },
};

// Worker-aware unary metrics (gRPC)
const grpcUnaryWorkerReqs = new Counter("grpc_unary_worker_reqs");
const grpcUnaryWorkerProcRssKb = new Trend("grpc_unary_worker_proc_rss_kb");
const grpcUnaryWorkerProcCpuPct = new Trend("grpc_unary_worker_proc_cpu_pct");
const grpcUnaryWorkersSeenLocal = new Trend("grpc_unary_workers_seen_local");
const grpcUnaryDbgWorkersSeen = new Gauge("grpc_unary_debug_workers_seen_last_batch");

export function setup() {
  grpc.loadModel("iris");
  return TestData;
}

export default function (data) {
  const modelName = __ENV.MODEL_NAME;

  const res = grpc.infer(data[modelName].grpc);

  const seenWorkers = new Set();

  try {
    const outputs = res.message?.outputs || [];
    const params = outputs[0]?.parameters || {};
    const headers = params.headers || {};
    const pid = headers["X-Worker-PID"] || "unknown";
    seenWorkers.add(pid);

    grpcUnaryWorkerReqs.add(1, { worker: pid, model_name: modelName });

    const rss = Number(headers["X-Proc-RSS-KB"]);
    if (!Number.isNaN(rss)) {
      grpcUnaryWorkerProcRssKb.add(rss, { worker: pid, model_name: modelName });
    }
    const cpu = Number(headers["X-Proc-CPU-Pct"]);
    if (!Number.isNaN(cpu)) {
      grpcUnaryWorkerProcCpuPct.add(cpu, { worker: pid, model_name: modelName });
    }
  } catch (_) {
    // ignore parse issues
  }

  grpcUnaryWorkersSeenLocal.add(seenWorkers.size, { model_name: modelName });
  grpcUnaryDbgWorkersSeen.add(seenWorkers.size, { model_name: modelName });
}

export function teardown(data) {
  grpc.unloadModel("iris");
}

/* ------------------------- Summary (gRPC unary) ------------------------- */

export function handleSummary(data) {
  const lines = [];
  lines.push("=== gRPC Unary Summary ===");
  lines.push(`Approx. distinct workers (max per-VU batch): ${approxWorkers}`);
  if (avgRss !== null) lines.push(`Average worker RSS (KB): ${avgRss.toFixed(0)}`);
  if (avgCpu !== null) lines.push(`Average worker CPU (%): ${avgCpu.toFixed(2)}`);
  lines.push("");

  // Classic k6 table
  const classic = textSummary(data, { indent: " ", enableColors: true });

  // Single stdout output containing both sections
  return { stdout: `${lines.join("\n")}\n${classic}` };
}
