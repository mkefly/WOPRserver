import os

import requests


class RestClient:
    def __init__(self, host=None, http_port=None, metrics_port=None):
        host = host or os.environ.get("MLSERVER_HOST", "0.0.0.0")
        http_port = http_port or os.environ.get("MLSERVER_HTTP_PORT", "8080")
        metrics_port = metrics_port or os.environ.get("MLSERVER_METRICS_PORT", http_port)
        self.rest_host = f"http://{host}:{http_port}"
        self.metrics_host = f"http://{host}:{metrics_port}"

    def metrics(self, session: requests.Session = None, timeout=3.0):
        sess = session or requests.Session()
        r = sess.get(f"{self.metrics_host}/metrics", timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text

    def load_model(self, name: str, session: requests.Session = None):
        sess = session or requests.Session()
        r = sess.post(f"{self.rest_host}/v2/repository/models/{name}/load")
        return r

    def unload_model(self, name: str, session: requests.Session = None):
        sess = session or requests.Session()
        r = sess.post(f"{self.rest_host}/v2/repository/models/{name}/unload")
        return r

    def infer(self, name: str, payload: dict, session: requests.Session = None):
        sess = session or requests.Session()
        r = sess.post(f"{self.rest_host}/v2/models/{name}/infer", json=payload, headers={"Content-Type": "application/json"})
        return r

    def infer_stream_begin(self, name: str, payload: dict, session: requests.Session = None):
        # returns a streaming Response object for SSE consumption
        sess = session or requests.Session()
        r = sess.post(f"{self.rest_host}/v2/models/{name}/infer_stream", json=payload, headers={"Content-Type": "application/json"}, stream=True)
        return r
