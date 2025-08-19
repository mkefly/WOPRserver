export function readTestData(name) {
  const base = import.meta.resolve(`../data/${name}/`);

  const readJson = (relPath) => {
    try {
      return JSON.parse(open(base + relPath));
    } catch (_) {
      return null;
    }
  };

  // single (existing)
  const restSingle = readJson("rest-requests.json");
  const grpcSingle = readJson("grpc-requests.json");

  // many (new, optional)
  const restMany = readJson("rest-requests-many.json"); // array | null
  const grpcMany = readJson("grpc-requests-many.json"); // array | null

  return {
    rest: restSingle,
    grpc: grpcSingle,
    rest_many: Array.isArray(restMany) ? restMany : null,
    grpc_many: Array.isArray(grpcMany) ? grpcMany : null,
  };
}
