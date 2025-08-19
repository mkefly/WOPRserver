from locust import HttpUser, task, between, events
import time, os, json, random
from sseclient import SSEClient
from common.helpers import read_test_data
from common.prom_utils import (
    count_active_workers, count_distinct_workers, sum_inference_pools, parse_prom_metric, parse_rest_stream_requests
)

TestData = {
    "summodel": read_test_data("summodel"),
    "llm": read_test_data("llm"),
}

def pick_random(arr):
    return random.choice(arr)

def _headers_from_chunk(data_str: str):
    try:
        msg = json.loads(data_str)
        outputs = msg.get("outputs", [])
        params = (outputs[0] or {}).get("parameters", {}) if outputs else {}
        headers = params.get("headers", {}) or {}
        return headers
    except Exception:
        return {}

class RestStreamUserBase(HttpUser):
    abstract = True  # <--- THIS LINE IS REQUIRED

    wait_time = between(0.1, 0.1)  # ~10Hz like k6 sleep(0.1)
    host = f"http://{os.environ.get('MLSERVER_HOST','127.0.0.1')}:{os.environ.get('MLSERVER_HTTP_PORT','8080')}"
    metrics_host = f"http://{os.environ.get('MLSERVER_HOST','127.0.0.1')}:{os.environ.get('MLSERVER_METRICS_PORT', os.environ.get('MLSERVER_HTTP_PORT','8080'))}"

    model_name = None

    def on_start(self):
        # Load both models once (cheap if already loaded)
        self.client.post("/v2/repository/models/summodel/load")
        self.client.post("/v2/repository/models/llm/load")

    def on_stop(self):
        self.client.post("/v2/repository/models/summodel/unload")
        self.client.post("/v2/repository/models/llm/unload")

    def _stream_once(self, payload):
        t0 = time.time()
        first_global = None
        first_seen_worker = {}
        seen_workers = set()

        with self.client.post(f"/v2/models/{self.model_name}/infer_stream", json=payload, stream=True, name=f"SSE /infer_stream {self.model_name}", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            sse = SSEClient(r)
            for evt in sse.events():
                events.request.fire(request_type="metric", name="sse_events", response_time=1.0, response_length=len(evt.data or ""), exception=None)
                if first_global is None:
                    first_global = time.time()
                    events.request.fire(request_type="metric", name="sse_first_event_ms", response_time=(first_global - t0)*1000.0, response_length=0, exception=None)
                headers = _headers_from_chunk(evt.data or "{}")
                pid = headers.get("X-Worker-PID", "unknown")
                seen_workers.add(pid)
                # per-worker stats
                if pid:
                    if pid not in first_seen_worker:
                        first_seen_worker[pid] = time.time()
                        events.request.fire(request_type="metric", name="worker_first_event_ms", response_time=(first_seen_worker[pid] - t0)*1000.0, response_length=0, context={"worker": pid}, exception=None)
                    # chunk size
                    events.request.fire(request_type="metric", name="worker_chunk_size_bytes", response_time=len(evt.data or ""), response_length=0, context={"worker": pid}, exception=None)
                    # total bytes via accumulating each chunk as a separate event
                    events.request.fire(request_type="metric", name="worker_bytes", response_time=len(evt.data or ""), response_length=0, context={"worker": pid}, exception=None)
                    # process telemetry
                    try:
                        rss = float(headers.get("X-Proc-RSS-KB", "nan"))
                        if rss == rss:  # not NaN
                            events.request.fire(request_type="metric", name="worker_proc_rss_kb", response_time=rss, response_length=0, context={"worker": pid}, exception=None)
                    except Exception: pass
                    try:
                        cpu = float(headers.get("X-Proc-CPU-Pct", "nan"))
                        if cpu == cpu:
                            events.request.fire(request_type="metric", name="worker_proc_cpu_pct", response_time=cpu, response_length=0, context={"worker": pid}, exception=None)
                    except Exception: pass

        dur_ms = (time.time() - t0) * 1000.0
        events.request.fire(request_type="metric", name="sse_total_duration_ms", response_time=dur_ms, response_length=0, exception=None)
        for pid in seen_workers:
            events.request.fire(request_type="metric", name="worker_stream_duration_ms", response_time=dur_ms, response_length=0, context={"worker": pid}, exception=None)
            events.request.fire(request_type="metric", name="worker_streams", response_time=1.0, response_length=0, context={"worker": pid}, exception=None)
        events.request.fire(request_type="metric", name="workers_seen_local", response_time=len(seen_workers), response_length=0, exception=None)

class RestStreamSumUser(RestStreamUserBase):
    model_name = "summodel"
    @task
    def stream_task(self):
        d = TestData[self.model_name]
        payload = random.choice(d["rest_many"]) if d["rest_many"] else d["rest"]
        self._stream_once(payload)

class RestStreamLLMUser(RestStreamUserBase):
    model_name = "llm"
    @task
    def stream_task(self):
        d = TestData[self.model_name]
        payload = random.choice(d["rest_many"]) if d["rest_many"] else d["rest"]
        self._stream_once(payload)

class PromPoller(HttpUser):
    wait_time = between(1.0, 1.0)
    host = f"http://{os.environ.get('MLSERVER_HOST','127.0.0.1')}:{os.environ.get('MLSERVER_METRICS_PORT', os.environ.get('MLSERVER_HTTP_PORT','8080'))}"

    @task
    def scrape(self):
        with self.client.get("/metrics", name="Prometheus /metrics", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.text
            active = count_active_workers(body)
            distinct = count_distinct_workers(body)
            pools = sum_inference_pools(body)
            loads = parse_prom_metric(body, 'worker_model_updates_total{outcome="success",type="Load"}')
            unloads = parse_prom_metric(body, 'worker_model_updates_total{outcome="success",type="Unload"}')
            req = parse_rest_stream_requests(body)

            if active is not None:
                events.request.fire(request_type="metric", name="rest_workers_active_prom", response_time=float(active), response_length=0, exception=None)
            if distinct is not None:
                events.request.fire(request_type="metric", name="rest_workers_distinct_prom", response_time=float(distinct), response_length=0, exception=None)
            if pools is not None:
                events.request.fire(request_type="metric", name="rest_pools_active_prom", response_time=float(pools), response_length=0, exception=None)
            if loads is not None:
                events.request.fire(request_type="metric", name="rest_models_loaded_total_prom", response_time=float(loads), response_length=0, exception=None)
            if unloads is not None:
                events.request.fire(request_type="metric", name="rest_models_unloaded_total_prom", response_time=float(unloads), response_length=0, exception=None)
            if req:
                events.request.fire(request_type="metric", name="rest_stream_reqs_200_prom", response_time=float(req["ok200"]), response_length=0, exception=None)
                events.request.fire(request_type="metric", name="rest_stream_reqs_other_prom", response_time=float(req["other"]), response_length=0, exception=None)
