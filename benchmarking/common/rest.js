import { check } from "k6";
import http from "k6/http";
// SSE support (xk6 extension). Build k6 with:
//   xk6 build --with github.com/phymbert/xk6-sse@latest
import sse from "k6/x/sse";

function checkResponse(res) {
  check(res, {
    "is status 200": (r) => r && r.status === 200,
  });
}

export class RestClient {
  constructor() {
    this.restHost = `http://${__ENV.MLSERVER_HOST}:${__ENV.MLSERVER_HTTP_PORT}`;
    this.metricsHost = `http://${__ENV.MLSERVER_HOST}:${__ENV.MLSERVER_METRICS_PORT || __ENV.MLSERVER_HTTP_PORT}`;
  }

  /**
   * Scrape Prometheus /metrics from MLServer (optional).
   * Returns string body on success, or null on failure.
   */
  metrics() {
    const url = `${this.metricsHost}/metrics`;
    const res = http.get(url, { timeout: "3s" });
    if (!res || res.status !== 200) return null;
    return res.body; // text/plain; version=0.0.4
  }

  loadModel(name) {
    const res = http.post(`${this.restHost}/v2/repository/models/${name}/load`);
    checkResponse(res);
    return res;
  }

  unloadModel(name) {
    const res = http.post(
      `${this.restHost}/v2/repository/models/${name}/unload`
    );
    checkResponse(res);
    return res;
  }

  infer(name, payload) {
    const headers = { "Content-Type": "application/json" };
    const res = http.post(
      `${this.restHost}/v2/models/${name}/infer`,
      JSON.stringify(payload),
      { headers }
    );
    checkResponse(res);
    return res;
  }

  /**
   * Server-Sent Events streaming call to MLServer's REST streaming endpoint.
   * MLServer: /v2/models/{model_name}/infer_stream (server-streaming)
   */
  inferStream(name, payload, onEvent, onError) {
    const url = `${this.restHost}/v2/models/${name}/infer_stream`;
    const params = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      tags: { protocol: "rest-stream", model_name: name },
    };

    const res = sse.open(url, params, (client) => {
      client.on("event", (event) => {
        if (onEvent) onEvent(event);
      });
      client.on("error", (e) => {
        if (onError) onError(e);
      });
    });

    checkResponse(res);
    return res;
  }
}
