import { check } from "k6";
import { Counter } from "k6/metrics";
import grpc from "k6/net/grpc";

/* -------------------- metrics -------------------- */
const grpcReqs = new Counter("grpc_reqs");
const grpcStreamMsgs = new Counter("grpc_stream_msgs");

/* -------------------- tiny path helpers -------------------- */
function toPosix(p) {
  return String(p || "").replace(/\\/g, "/");
}
function fileURLToPath(u) {
  const s = String(u || "");
  return s.startsWith("file://") ? s.replace(/^file:\/\//, "") : s;
}
function splitPath(p) {
  const s = toPosix(p);
  const i = s.lastIndexOf("/");
  return i >= 0 ? { dir: s.slice(0, i), base: s.slice(i + 1) } : { dir: ".", base: s };
}

/**
 * Resolve dataplane.proto:
 * 1) MLSERVER_PROTO (absolute file path strongly recommended)
 * 2) benchmarking/proto/dataplane.proto   (relative to this module)
 * 3) proto/dataplane.proto                (one level up)
 * Else: throw a helpful error.
 */
function resolveProtoAbsPath() {
  // 1) explicit env override
  if (__ENV.MLSERVER_PROTO && __ENV.MLSERVER_PROTO.trim()) {
    return toPosix(__ENV.MLSERVER_PROTO.trim());
  }

  // 2) ./../proto/dataplane.proto (module-relative: common/ -> benchmarking/proto/)
  try {
    const url = import.meta.resolve("../proto/dataplane.proto");
    return fileURLToPath(url);
  } catch (_) {
    // continue
  }

  // 3) ./../../proto/dataplane.proto (module-relative: common/ -> proto/)
  try {
    const url = import.meta.resolve("../../proto/dataplane.proto");
    return fileURLToPath(url);
  } catch (_) {
    // continue
  }

  // Give a clear instruction
  throw new Error(
    [
      "Could not locate dataplane.proto.",
      "",
      "Fix by providing one of these:",
      "  1) Pass an absolute path via env: -e MLSERVER_PROTO=/abs/path/to/dataplane.proto",
      "  2) Place a copy at: benchmarking/proto/dataplane.proto",
      "  3) Or at:           proto/dataplane.proto",
      "",
      "Tip: in your Poetry venv it’s often under:",
      "  <venv>/lib/python*/site-packages/mlserver/grpc/protos/dataplane.proto",
    ].join("\n")
  );
}

function getClient() {
  const client = new grpc.Client();

  // IMPORTANT: pass dir in imports[], and ONLY the basename as proto file
  const protoAbs = resolveProtoAbsPath();
  const { dir, base } = splitPath(protoAbs);
  client.load([dir], base);

  return client;
}

function checkUnaryResponse(res) {
  check(res, {
    "status is OK": (r) => r && r.status === grpc.StatusOK,
  });
  grpcReqs.add(1);
}

/* -------------------- exported client -------------------- */
export class GrpcClient {
  constructor() {
    const host = __ENV.MLSERVER_HOST || "127.0.0.1";
    const port = __ENV.MLSERVER_GRPC_PORT || "8081";
    this.grpcHost = `${host}:${port}`;
    this.client = getClient();
    this.connected = false;
  }

  connect() {
    if (!this.connected) {
      // plaintext is expected for local benches
      this.client.connect(this.grpcHost, { plaintext: true });
      this.connected = true;
    }
  }

  loadModel(name) {
    this.connect();
    const payload = { model_name: name };
    const res = this.client.invoke(
      "inference.GRPCInferenceService/RepositoryModelLoad",
      payload
    );
    checkUnaryResponse(res);
  }

  unloadModel(name) {
    this.connect();
    const payload = { model_name: name };
    const res = this.client.invoke(
      "inference.GRPCInferenceService/RepositoryModelUnload",
      payload
    );
    checkUnaryResponse(res);
  }

  /** Unary inference. Returns response so callers can parse headers/outputs. */
  infer(payload) {
    this.connect();
    const res = this.client.invoke(
      "inference.GRPCInferenceService/ModelInfer",
      payload
    );
    checkUnaryResponse(res);
    return res;
  }

  /**
   * Server/bidi streaming via ModelStreamInfer.
   * Sends one request (server-streaming); write more for bidi patterns if needed.
   */
  streamInfer(payload, handlers = {}) {
    this.connect();

    const stream = new grpc.Stream(
      this.client,
      "inference.GRPCInferenceService/ModelStreamInfer",
      null
    );

    stream.on("data", (message) => {
      grpcStreamMsgs.add(1);
      if (handlers.onData) handlers.onData(message);
    });
    stream.on("end", () => {
      if (handlers.onEnd) handlers.onEnd();
    });
    stream.on("error", (e) => {
      if (handlers.onError) handlers.onError(e);
    });

    // one request (server streaming)
    stream.write(payload);
    stream.end();
  }

  close() {
    this.client.close();
  }
}
