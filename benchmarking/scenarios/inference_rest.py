import json
import os

from common.helpers import read_test_data
from locust import HttpUser, between, events, task

# Config (mirror k6)
SCENARIO_DURATION = "60s"  # for docs
SCENARIO_VUS = 300         # run via CLI: -u 300 -t 60s

TestData = {
    "iris": read_test_data("iris"),
}

def _parse_worker_headers_from_json_body(text):
    try:
        body = json.loads(text)
        outputs = body.get("outputs", [])
        params = (outputs[0] or {}).get("parameters", {}) if outputs else {}
        headers = params.get("headers", {}) or {}
        return headers
    except Exception:
        return {}

class RestIrisUser(HttpUser):
    wait_time = between(0.0, 0.0)
    host = f"http://{os.environ.get('MLSERVER_HOST','127.0.0.1')}:{os.environ.get('MLSERVER_HTTP_PORT','8080')}"

    def on_start(self):
        # Best-effort model load
        self.client.post("/v2/repository/models/iris/load")

    def on_stop(self):
        self.client.post("/v2/repository/models/iris/unload")

    @task
    def iris_infer(self):
        payload = TestData["iris"]["rest"]
        with self.client.post("/v2/models/iris/infer", json=payload, name="REST /infer iris", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            headers = _parse_worker_headers_from_json_body(r.text)
            pid = headers.get("X-Worker-PID", "unknown")
            # Emit custom "per-worker" metrics by piggybacking on request events:
            if "X-Proc-RSS-KB" in headers:
                try:
                    rss = float(headers["X-Proc-RSS-KB"])
                    events.request.fire(request_type="metric", name="unary_worker_proc_rss_kb", response_time=rss, response_length=0, context={"worker": pid}, exception=None)
                except Exception:
                    pass
            if "X-Proc-CPU-Pct" in headers:
                try:
                    cpu = float(headers["X-Proc-CPU-Pct"])
                    events.request.fire(request_type="metric", name="unary_worker_proc_cpu_pct", response_time=cpu, response_length=0, context={"worker": pid}, exception=None)
                except Exception:
                    pass
            # Distinct workers seen per-request (will usually be 1 for unary)
            events.request.fire(request_type="metric", name="unary_workers_seen_local", response_time=1.0, response_length=0, context={"workers": 1}, exception=None)
