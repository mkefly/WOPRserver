import asyncio
import multiprocessing as mp
from asyncio import Task
from multiprocessing import Queue

from mlserver.settings import Settings
from multiprocessing.context import BaseContext

from ..logging import get_logger 
logger = get_logger()


__all__ = [
    "END_OF_QUEUE",
    "cancel_task",
    "configure_inference_pool",
    "get_mp_context",
    "make_queue",
    "terminate_queue",
]

# Use a unique object for the sentinel to avoid clashing with legitimate None payloads.
END_OF_QUEUE: object = object()


def configure_inference_pool(settings: Settings) -> None:
    """
    Configure multiprocessing for the inference pool.

    - Forces the 'spawn' start method when parallel workers are enabled (idempotent).
    - Logs (rather than crashes) if the start method was already set elsewhere.
    - Safe to call multiple times; only effective on the first set in a process.
    """
    if not settings.parallel_workers:
        return

    try:
        current = mp.get_start_method(allow_none=True)
    except TypeError:
        # Python <3.8 compat: get_start_method has no allow_none kwarg
        try:
            current = mp.get_start_method()
        except RuntimeError:
            current = None

    if current == "spawn":
        logger.debug("Multiprocessing start method already set to 'spawn'.")
        return

    try:
        # force=True so tests (e.g., pytest using fork) don't leak the wrong context
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # The context was already set; keep going but warn.
        method = None
        try:
            method = mp.get_start_method()
        except Exception:
            pass
        logger.warning(
            "Unable to set multiprocessing start method to 'spawn' (already initialized). "
            f"Current method is '{method}'. Ensure 'spawn' is used for worker processes."
        )


def get_mp_context() -> BaseContext:
    """Return the multiprocessing context to use for shared objects (defaults to 'spawn')."""
    try:
        method = mp.get_start_method()
    except RuntimeError:
        method = "spawn"
    return mp.get_context(method)


def make_queue(maxsize: int = 0) -> Queue:
    """
    Create a multiprocessing.Queue from the current context (preferably 'spawn').

    Using the context ensures the queue is compatible with the chosen start method.
    """
    ctx = mp.get_context()
    q = ctx.Queue(maxsize=maxsize)
    # Optional: avoid join thread waits on shutdown
    try:
        q.cancel_join_thread()
    except Exception:
        pass
    return q

async def terminate_queue(queue: Queue, *, timeout: float = 0.0) -> bool:
    """
    Best-effort: send the sentinel to a multiprocessing.Queue to signal termination.

    Returns:
        True if the sentinel was put (or the queue looked already closed),
        False if a put attempt definitively failed.

    Notes:
    - mp.Queue.put can block; we use run_in_executor to avoid blocking the event loop.
    - If the queue is already closed or broken, we swallow the error and return True.
    """
    loop = asyncio.get_running_loop()

    def _put() -> bool:
        try:
            if timeout and timeout > 0:
                queue.put(END_OF_QUEUE, timeout=timeout)
            else:
                # mp.Queue has no put_nowait; emulate with block=False
                queue.put(END_OF_QUEUE, block=False)
            return True
        except (ValueError, AssertionError, OSError):
            # Queue already closed/broken or cannot accept more items
            return True
        except Exception:
            logger.exception("Failed to put END_OF_QUEUE into multiprocessing Queue.")
            return False

    return await loop.run_in_executor(None, _put)


async def cancel_task(task: Task | None, *, timeout: float | None = None) -> None:
    """
    Cancel an asyncio Task safely.

    Args:
        task: The task to cancel (ignored if None or already done).
        timeout: Optional timeout to await task completion after cancel.
    """
    if task is None or task.done():
        return

    task.cancel()
    try:
        if timeout is None:
            await task
        else:
            await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        # Expected when cancellation propagates
        pass
    except asyncio.TimeoutError:
        # The task didn't finish in time; it will continue to be cancelled in the background.
        logger.debug("Timed out while awaiting task cancellation.")
    except Exception:
        # Defensive: tasks may raise during cancellation; log and continue shutdown.
        logger.exception("Task raised an exception during cancellation.")
