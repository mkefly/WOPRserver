from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from collections.abc import AsyncIterator, Iterable
from typing import Union, Optional

from mlserver import MLModel, types
from woprserver.logging import get_logger

logger = get_logger()

# =============================================================================
# Tunables (override via env)
# =============================================================================

TOKENS_PER_CHUNK = int(os.getenv("TOKENS_PER_CHUNK", "36"))  # ~20–40 is reasonable
TIMESLICE_MS = int(os.getenv("TIMESLICE_MS", "90"))          # ~50–100 ms typical

MIN_TOKENS = int(os.getenv("MIN_TOKENS", "400"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1600"))

HEADING_EVERY_N_PARAS = 4
HEADING_PROB = 0.40        # chance to emit an initial heading
SECTION_PROB = 0.70        # chance to emit a section heading at cadence

BULLETS_CHANCE = 0.18
MAX_BULLETS = 5

SENTENCE_MIN_WORDS = 6
SENTENCE_MAX_WORDS = 22

PARA_MIN_SENTENCES = 2
PARA_MAX_SENTENCES = 6


# =============================================================================
# Helpers
# =============================================================================

ByteLike = Union[bytes, bytearray, memoryview]


def _seed_from_request(req_id: Optional[str], prompt: str) -> int:
    """Deterministic per request + prompt."""
    h = hashlib.blake2b(digest_size=8)
    h.update((req_id or "none").encode("utf-8"))
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def _normalize_prompt_words(prompt: str) -> list[str]:
    return [
        w.strip().strip(",.!?;:()[]{}\"'").lower()
        for w in prompt.split()
        if w.strip()
    ]


def _base_vocab() -> list[str]:
    base = (
        "the of to and a in that is for on with as it at by from be are was were "
        "not this or an can if which more one all about will into use learn model "
        "data time system request stream response token latency parallel worker "
        "throughput cache batching network backpressure pipeline chunk scheduler "
        "async await coroutine process thread queue event loop server client api "
        "http grpc sse json gzip compression rate limit performance reliability"
    ).split()
    # de-dup, stable order
    return list(dict.fromkeys(base))


def _title_case(words: list[str]) -> str:
    return " ".join(w.capitalize() for w in words if w).strip()


def _choose_heading(prompt_words: list[str], rng: random.Random) -> str:
    picks = prompt_words or _base_vocab()
    length = rng.randint(2, 5)
    return _title_case([rng.choice(picks) for _ in range(length)])


def _word_sampler(prompt: str, rng: random.Random) -> Iterable[str]:
    """
    Topic-biased word stream: mixes prompt words with base vocab.
    Occasionally injects connective phrases to improve flow.
    """
    prompt_words = [w for w in _normalize_prompt_words(prompt) if len(w) > 2]
    base = _base_vocab()
    vocab = prompt_words * 4 + base  # heavy bias toward prompt
    connectors = [
        "moreover", "furthermore", "consequently", "however", "therefore",
        "interestingly", "importantly", "notably", "in practice", "in short",
        "as a result", "meanwhile", "in parallel", "on the other hand",
    ]
    while True:
        if rng.random() < 0.06:
            yield rng.choice(connectors)
        yield rng.choice(vocab)


def _safe_sample(rng: random.Random, population: list[int], k: int) -> list[int]:
    """Sample up to k items safely (guards against k > len(population))."""
    k = max(0, min(k, len(population)))
    if k == 0:
        return []
    return rng.sample(population, k)


def _sentence(words: Iterable[str], rng: random.Random) -> str:
    n = rng.randint(SENTENCE_MIN_WORDS, SENTENCE_MAX_WORDS)

    toks: list[str] = []
    # choose potential comma positions between 3 and n-2 (exclusive of end)
    start = 3
    stop = max(4, n - 2)  # ensure at least one candidate when n is small
    candidates = list(range(start, stop)) if stop > start else []
    append_after = set(_safe_sample(rng, candidates, rng.randint(0, 2)))

    for i in range(n):
        w = next(words)
        if i == 0:
            w = w.capitalize()
        if i in append_after and rng.random() < 0.6:
            toks.append(w + ",")
        else:
            toks.append(w)

    end = rng.choices([".", ".", ".", ".", "!", "?"], weights=[5, 1, 1, 1, 0.3, 0.2])[0]
    return " ".join(toks) + end


def _paragraph(words: Iterable[str], rng: random.Random) -> str:
    s = rng.randint(PARA_MIN_SENTENCES, PARA_MAX_SENTENCES)
    return " ".join(_sentence(words, rng) for _ in range(s))


def _bullets(words: Iterable[str], rng: random.Random, count: int) -> list[str]:
    count = max(1, min(count, MAX_BULLETS))
    bullets = []
    for _ in range(count):
        n = rng.randint(5, 14)
        toks = [next(words) for _ in range(n)]
        bullets.append("- " + toks[0].capitalize() + " " + " ".join(toks[1:]) + ".")
    return bullets


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _join_and_cleanup(chunks: list[str]) -> str:
    text = " ".join(chunks)
    # small fixes to spacing/punctuation
    return (
        text.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
    )


# ============================ IO: decode/encode ==============================

def _decode_first_cell(req: types.InferenceRequest) -> str:
    """
    Return the first input element as a *text* string.
    Handles bytes/bytearray/memoryview safely; falls back to str().
    """
    if not getattr(req, "inputs", None):
        return ""
    first = req.inputs[0]
    data = getattr(first, "data", None)
    if not data:
        return ""
    # mlserver supports both "list-like" or TensorData(root=[...])
    try:
        # TensorData
        root = data.root  # type: ignore[attr-defined]
        x = root[0] if root else b""
    except AttributeError:
        # Plain list-like
        x = data[0]
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def _to_bytes(s: str) -> bytes:
    return s.encode("utf-8", errors="strict")


def _make_text_output_bytes(name: str, payload: str, headers: dict[str, str]) -> types.ResponseOutput:
    return types.ResponseOutput(
        name=name,
        shape=[1],
        datatype="BYTES",
        data=types.TensorData(root=[_to_bytes(payload)]),
        parameters=types.Parameters(headers=headers),
    )


# =============================================================================
# Model
# =============================================================================

class ToyLLM(MLModel):
    """
    A toy model that streams long but readable prose with sections and lists.
    It coalesces tokens by count and by time slice to keep SSE event rate low.

    Robust to MLServer delivering *batches* to predict/predict_stream.
    Accepts:
      - a single `types.InferenceRequest`
      - a list/tuple of `types.InferenceRequest`
      - a (sync) iterator of `types.InferenceRequest`
      - an async iterator of `types.InferenceRequest`
    Uses the first request as the driver of the response.
    """

    # --- helpers for request handling -------------------------------------
    async def _first_request(self, payload_or_batch) -> types.InferenceRequest:
        """Return the first InferenceRequest from common batch shapes."""
        # Single request
        if isinstance(payload_or_batch, types.InferenceRequest):
            return payload_or_batch

        # List / tuple
        if isinstance(payload_or_batch, (list, tuple)):
            if not payload_or_batch:
                raise ValueError("Empty batch for predict/predict_stream")
            return payload_or_batch[0]

        # Sync iterator
        try:
            iterator = iter(payload_or_batch)  # type: ignore[arg-type]
        except TypeError:
            iterator = None
        if iterator is not None:
            try:
                return next(iterator)
            except StopIteration:
                raise ValueError("Empty iterator for predict/predict_stream")

        # Async iterator
        try:
            aiter = payload_or_batch.__aiter__()  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            try:
                return await aiter.__anext__()    # type: ignore[attr-defined]
            except StopAsyncIteration:
                raise ValueError("Empty async iterator for predict/predict_stream")

        raise TypeError("Unsupported payload type for predict/predict_stream")

    def _get_prompt(self, req: types.InferenceRequest) -> str:
        return _decode_first_cell(req)

    async def _generate_stream(
        self,
        prompt: str,
        min_tokens: int = MIN_TOKENS,
        max_tokens: int = MAX_TOKENS,
        req_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Yields *semantic* chunks (paras, bullet blocks, headings), not single tokens.
        A token budget controls overall length; callers can still coalesce.
        """
        rng = random.Random(_seed_from_request(req_id, prompt))
        target_tokens = rng.randint(min_tokens, max_tokens)

        # Simulate prefill + tiny overhead
        await asyncio.sleep(rng.uniform(0.03, 0.12))
        await asyncio.sleep(0.01)

        words = _word_sampler(prompt, rng)
        tokens_emitted = 0
        paras = 0

        # Optional initial heading
        if rng.random() < HEADING_PROB:
            heading = _choose_heading(_normalize_prompt_words(prompt), rng)
            yield f"## {heading}\n"
            tokens_emitted += len(heading.split())
            paras += 1

        while tokens_emitted < target_tokens:
            try:
                if rng.random() < BULLETS_CHANCE:
                    count = rng.randint(2, MAX_BULLETS)
                    bullets = _bullets(words, rng, count)
                    bl_text = "\n".join(bullets) + "\n"
                    yield bl_text
                    tokens_emitted += sum(len(b.split()) for b in bullets)
                else:
                    para = _paragraph(words, rng)
                    yield para + "\n\n"
                    tokens_emitted += len(para.split())
                    paras += 1

                # Occasional section heading between paragraphs
                if (
                    paras > 0
                    and paras % HEADING_EVERY_N_PARAS == 0
                    and rng.random() < SECTION_PROB
                ):
                    heading = _choose_heading(_normalize_prompt_words(prompt), rng)
                    yield f"### {heading}\n"

                # Short think time between logical chunks
                await asyncio.sleep(rng.uniform(0.006, 0.02))
            except Exception:
                # Defensive: skip malformed block and continue
                continue

    # ------------------------------------------------------------------
    # Inference APIs
    # ------------------------------------------------------------------

    async def predict_stream(self, payload):
        """
        Streaming response with coalescing:
          - buffer by TOKENS_PER_CHUNK
          - flush at least every TIMESLICE_MS
        Emits fewer SSE events while preserving low TTFT.

        Accepts either a single InferenceRequest or a batch/iterator.
        """
        pid = str(os.getpid())
        req = await self._first_request(payload)
        rid = req.id or "unknown"
        prompt = self._get_prompt(req)

        t0 = time.time()
        logger.info(f"LLM START pid={pid} req={rid} t={t0:.6f}")

        buffer: list[str] = []
        chunk_idx = 0
        last_emit = _now_ms()

        def _buffer_token_count() -> int:
            # Rough word-based “token” proxy
            return sum(len(p.split()) for p in buffer)

        async def _flush() -> Optional[types.InferenceResponse]:
            nonlocal buffer, chunk_idx, last_emit
            if not buffer:
                return None
            text = _join_and_cleanup(buffer)
            buffer.clear()
            last_emit = _now_ms()
            chunk_idx += 1

            out = _make_text_output_bytes(
                name="text",
                payload=text,
                headers={
                    "X-Worker-PID": pid,
                    "X-Chunk-Index": str(chunk_idx),
                    "X-Coalesce": f"{TOKENS_PER_CHUNK}/{TIMESLICE_MS}ms",
                },
            )
            return types.InferenceResponse(
                id=req.id,
                model_name=self.name,
                outputs=[out],
                parameters=types.Parameters(headers={"Ce-Endpoint": "llm"}),
            )

        try:
            async for piece in self._generate_stream(prompt, req_id=req.id):
                buffer.append(piece)

                time_due = (_now_ms() - last_emit) >= TIMESLICE_MS
                count_due = _buffer_token_count() >= TOKENS_PER_CHUNK

                if time_due or count_due:
                    resp = await _flush()
                    if resp is not None:
                        yield resp

            # final flush
            resp = await _flush()
            if resp is not None:
                yield resp

        finally:
            t1 = time.time()
            logger.info(
                f"LLM END   pid={pid} req={rid} t={t1:.6f} dur={t1-t0:.3f}s chunks={chunk_idx}"
            )

    async def predict(self, payload) -> types.InferenceResponse:
        """
        Non-streaming endpoint: returns the complete text.
        Accepts either a single InferenceRequest or a batch/iterator.
        """
        pid = str(os.getpid())
        req = await self._first_request(payload)
        prompt = self._get_prompt(req)
        text = _join_and_cleanup(
            [chunk async for chunk in self._generate_stream(prompt, req_id=req.id)]
        )

        out = _make_text_output_bytes(
            name="text",
            payload=text,
            headers={"X-Worker-PID": pid},
        )
        return types.InferenceResponse(
            id=req.id,
            model_name=self.name,
            outputs=[out],
            parameters=types.Parameters(headers={"Ce-Endpoint": "llm"}),
        )
