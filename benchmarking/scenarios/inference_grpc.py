import time

from common.grpc_client import GrpcClient
from common.helpers import read_test_data
from locust import User, between, events, task

SCENARIO_DURATION = "60s"
SCENARIO_VUS = 300

TestData = {
    "iris": read_test_data("iris"),
}

class GrpcIrisUser(User):
    wait_time = between(0.0, 0.0)

    def on_start(self):
        self.client_grpc = GrpcClient()
        try:
            self.client_grpc.load_model("iris")
        except Exception:
            pass

    def on_stop(self):
        try:
            self.client_grpc.unload_model("iris")
        except Exception:
            pass
        self.client_grpc.close()

    @task
    def iris_infer(self):
        payload = TestData["iris"]["grpc"]
        t0 = time.time()
        ok = True
        exc = None
        try:
            res = self.client_grpc.infer(payload)
        except Exception as e:
            ok = False
            exc = e
        dt_ms = (time.time() - t0) * 1000.0
        events.request.fire(request_type="gRPC", name="ModelInfer iris", response_time=dt_ms, response_length=0, exception=None if ok else exc)

        if ok:
            # Extract headers from outputs[0].parameters.headers if present
            try:
                outputs = res.outputs or []
                params = outputs[0].parameters if outputs else None
                headers = dict(params.headers) if params and hasattr(params, "headers") else {}
                pid = headers.get("X-Worker-PID", "unknown")
                if "X-Proc-RSS-KB" in headers:
                    events.request.fire(request_type="metric", name="grpc_unary_worker_proc_rss_kb", response_time=float(headers["X-Proc-RSS-KB"]), response_length=0, context={"worker": pid}, exception=None)
                if "X-Proc-CPU-Pct" in headers:
                    events.request.fire(request_type="metric", name="grpc_unary_worker_proc_cpu_pct", response_time=float(headers["X-Proc-CPU-Pct"]), response_length=0, context={"worker": pid}, exception=None)
                events.request.fire(request_type="metric", name="grpc_unary_workers_seen_local", response_time=1.0, response_length=0, context={"workers": 1}, exception=None)
            except Exception:
                pass
