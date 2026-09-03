"""
Progress bar management for multiprocessing benchmarks.

This module provides a centralized progress tracking system using a listener
thread that aggregates progress updates from multiple worker processes via
a multiprocessing queue.
"""

import threading
from enum import Enum
from multiprocessing import Queue
from queue import Empty
from typing import NamedTuple

from tqdm import tqdm

# Sentinel value to signal listener shutdown
_SHUTDOWN_SENTINEL = ("__shutdown__", 0)

# Timeout for queue.get() to allow periodic liveness checks
_QUEUE_TIMEOUT_SECONDS = 5.0


class ProgressChannel(str, Enum):
    """
    The nesting levels a worker can report progress on.

    Each member's value is the literal that travels over the queue.

    Examples:
        >>> from hgp_lib.benchmarkers.progress import ProgressChannel
        >>> ProgressChannel.FOLD.value
        'fold'
    """

    EPOCH = "epoch"
    FOLD = "fold"
    RUN = "run"


class ProgressConfig(NamedTuple):
    """Configuration for progress tracking in multiprocessing benchmarks."""

    total_runs: int
    total_folds: int
    total_epochs: int
    show_run_progress: bool = True
    show_fold_progress: bool = True
    show_epoch_progress: bool = True


class ProgressListener:
    """
    Listener thread for aggregating progress updates from worker processes.

    Runs in the main process and listens to a multiprocessing queue for
    progress updates from worker processes. Updates three tqdm progress bars:
    - Experiments (runs)
    - Folds
    - Epochs

    The listener uses timeouts on queue.get() to periodically check for stop
    signals, preventing indefinite hangs if workers crash.

    Args:
        progress_queue: Multiprocessing queue for receiving progress updates.
        config: Progress configuration with totals and display settings.

    Examples:
        >>> from multiprocessing import Queue
        >>> from hgp_lib.benchmarkers.progress import ProgressConfig, ProgressListener
        >>> q = Queue()
        >>> cfg = ProgressConfig(
        ...     total_runs=1, total_folds=1, total_epochs=1,
        ...     show_run_progress=False, show_fold_progress=False,
        ...     show_epoch_progress=False,
        ... )
        >>> listener = ProgressListener(q, cfg)
        >>> listener.start()
        >>> q.put(("epoch", 1))
        >>> q.put(("fold", 1))
        >>> q.put(("run", 1))
        >>> listener.join()
    """

    def __init__(self, progress_queue: Queue, config: ProgressConfig):
        self.queue = progress_queue
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pbar_exp: tqdm | None = None
        self._pbar_fold: tqdm | None = None
        self._pbar_epoch: tqdm | None = None

    def start(self) -> None:
        """Start the listener thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait for it to finish."""
        self._stop_event.set()
        # Send sentinel to unblock queue.get() if waiting
        try:
            self.queue.put(_SHUTDOWN_SENTINEL)
        except (BrokenPipeError, EOFError):
            pass  # Queue already closed
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def join(self) -> None:
        """Wait for the listener thread to finish naturally (all runs completed)."""
        if self._thread is not None:
            self._thread.join()

    def _listen(self) -> None:
        """
        Main listener loop. Uses timeout to allow periodic stop checks.

        Expected message format: ``(channel, count)``, where ``channel`` is a
        :class:`ProgressChannel` value. The matching bar advances by ``count``.
        Unknown channels are ignored.
        """
        # Initialize progress bars (position=0 is bottom, position=2 is top)
        self._pbar_exp = tqdm(
            total=self.config.total_runs,
            position=0,
            desc="Runs",
            leave=True,
            disable=not self.config.show_run_progress,
        )
        self._pbar_fold = tqdm(
            total=self.config.total_folds,
            position=1,
            desc="Folds",
            leave=True,
            disable=not self.config.show_fold_progress,
        )
        self._pbar_epoch = tqdm(
            total=self.config.total_epochs,
            position=2,
            desc="Epochs",
            leave=True,
            disable=not self.config.show_epoch_progress,
        )

        bars = {
            ProgressChannel.EPOCH.value: self._pbar_epoch,
            ProgressChannel.FOLD.value: self._pbar_fold,
            ProgressChannel.RUN.value: self._pbar_exp,
        }

        finished_runs = 0

        try:
            while finished_runs < self.config.total_runs:
                # Check if stop was requested
                if self._stop_event.is_set():
                    break

                try:
                    msg, count = self.queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
                except Empty:
                    # Timeout - loop back to check stop_event and continue waiting
                    continue

                # Check for shutdown sentinel
                if msg == _SHUTDOWN_SENTINEL[0]:
                    break

                pbar = bars.get(msg)
                if pbar is None:
                    continue  # Unknown channel

                pbar.update(count)
                if msg == ProgressChannel.RUN.value:
                    finished_runs += count
        finally:
            # Ensure bars are closed properly
            self._pbar_epoch.close()
            self._pbar_fold.close()
            self._pbar_exp.close()


class ProgressReporter:
    """
    Producer-side handle for reporting progress, one method per channel.

    Wraps the queue that :class:`ProgressListener` consumes. A reporter built
    without a queue is inert and discards every update.

    Every method has the shape ``Callable[[int], None]``, so a bound method such
    as ``reporter.epoch`` can be passed as ``TrainerConfig.progress_callback``.

    Args:
        progress_queue (Queue | None):
            Queue to publish updates on, or ``None`` for an inert reporter.
            Default: `None`.

    Examples:
        >>> from multiprocessing import Queue
        >>> from hgp_lib.benchmarkers.progress import ProgressReporter
        >>> reporter = ProgressReporter(Queue())
        >>> reporter.enabled
        True
        >>> reporter.fold()
        >>> reporter.queue.get()
        ('fold', 1)
        >>> reporter.epoch(10)
        >>> reporter.queue.get()
        ('epoch', 10)

        An inert reporter accepts the same calls and discards them:

        >>> inert = ProgressReporter()
        >>> inert.enabled
        False
        >>> inert.run(5)
    """

    # "Queue | None" because type[Queue] is <method>..
    def __init__(self, progress_queue: "Queue | None" = None):
        self.queue = progress_queue

    @property
    def enabled(self) -> bool:
        """bool: Whether updates are published rather than discarded."""
        return self.queue is not None

    def epoch(self, count: int = 1) -> None:
        """Report ``count`` completed epochs. Default: `1`."""
        self._send(ProgressChannel.EPOCH, count)

    def fold(self, count: int = 1) -> None:
        """Report ``count`` completed folds. Default: `1`."""
        self._send(ProgressChannel.FOLD, count)

    def run(self, count: int = 1) -> None:
        """Report ``count`` completed runs. Default: `1`."""
        self._send(ProgressChannel.RUN, count)

    def _send(self, channel: ProgressChannel, count: int) -> None:
        """Publish ``(channel, count)`` unless this reporter is inert."""
        if self.queue is not None:
            self.queue.put((channel.value, count))
