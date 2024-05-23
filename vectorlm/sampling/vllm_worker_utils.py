from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper,
    ResultHandler,
    _run_worker_process,
    mp,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from multiprocessing import Queue
    from multiprocessing.process import BaseProcess

logger = init_logger(__name__)
JOIN_TIMEOUT_S = 2


class ManagedProcessWorkerWrapper(ProcessWorkerWrapper):
    """Wrap ProcessWorkerWrapper to add vectorlm thread to vllm process."""

    def __init__(
        self,
        result_handler: ResultHandler,
        worker_factory: Callable[[], Any],
        vectorlm_fn: Callable[[], None],
    ) -> None:
        """Initialize multiprocessing queues and launch worker process."""
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks

        self.process: BaseProcess = mp.Process(  # type: ignore[attr-defined]
            target=_run_worker_process_and_vectorlm_thread,
            name="VllmWorkerProcess",
            kwargs={
                "worker_factory": worker_factory,
                "task_queue": self._task_queue,
                "result_queue": self.result_queue,
                "vectorlm_fn": vectorlm_fn,
            },
            daemon=True,
        )

        self.process.start()


def _run_worker_process_and_vectorlm_thread(
    worker_factory: Callable[[], Any],
    task_queue: Queue,
    result_queue: Queue,
    vectorlm_fn: Callable[[], None],
) -> None:
    """Invoke _run_worker_process and vectorlm logic in separate thread."""
    # Add process-specific prefix to stdout and stderr

    vectorlm_thread = threading.Thread(target=vectorlm_fn)
    vectorlm_thread.start()

    _run_worker_process(worker_factory, task_queue, result_queue)
