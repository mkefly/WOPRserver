import re
from typing import Optional, Dict

def escape_re(s: str) -> str:
    return re.escape(s)

def parse_prom_metric(body: str, exact_line_prefix: str) -> Optional[float]:
    m = re.search(rf"^{re.escape(exact_line_prefix)}\s+([0-9eE+\-.]+)\s*$", body, re.M)
    return float(m.group(1)) if m else None

def count_distinct_workers(body: str) -> Optional[int]:
    if not body:
        return None
    setp = set(re.findall(r'^worker_active_streams\{[^}]*pid="([^"]+)"[^}]*\}\s+[0-9eE+\-.]+\s*$', body, re.M))
    return len(setp) or None

def count_active_workers(body: str) -> Optional[int]:
    if not body:
        return None
    active = 0
    for pid, val in re.findall(r'^worker_active_streams\{[^}]*pid="([^"]+)"[^}]*\}\s+([0-9eE+\-.]+)\s*$', body, re.M):
        try:
            v = float(val)
            if v > 0:
                active += 1
        except Exception:
            pass
    return active

def sum_inference_pools(body: str) -> Optional[float]:
    if not body:
        return None
    vals = re.findall(r'^inference_pools_active\{[^}]*pid="[^"]+"[^}]*\}\s+([0-9eE+\-.]+)\s*$', body, re.M)
    if not vals:
        return None
    s = 0.0
    for v in vals:
        try:
            s += float(v)
        except Exception:
            pass
    return s

def parse_labels(label_str: str) -> Dict[str, str]:
    return dict(re.findall(r'(\\w+)="([^"]*)"', label_str))

def parse_rest_stream_requests(body: str):
    if not body:
        return None
    ok200 = 0.0
    other = 0.0
    saw = False
    for labels, val in re.findall(r'^rest_server_requests_total\{([^}]*)\}\s+([0-9eE+\-.]+)\s*$', body, re.M):
        try:
            v = float(val)
        except Exception:
            continue
        lab = parse_labels(labels)
        path = lab.get("path", "")
        if path.startswith("/v2/models/") and "/infer_stream" in path:
            sc = lab.get("status_code", "")
            if sc == "200":
                ok200 += v
            else:
                other += v
            saw = True
    return {"ok200": ok200, "other": other, "total": ok200 + other} if saw else None
