"""
Progress bar management for multiprocessing benchmarks.

This module provides a centralized progress tracking system using a listener
thread that aggregates progress updates from multiple worker processes via
a multiprocessing queue.
"""

import queue
import threading
from multiprocessing import Queue
from typing import NamedTuple, Optional

from tqdm import tqdm


# Sentinel value to signal listener shutdown
_SHUTDOWN_SENTINEL = ("__shutdown__", 0)

# Timeout for queue.get() to allow periodic liveness checks
_QUEUE_TIMEOUT_SECONDS = 5.0


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

    Example:
        >>> from multiprocessing import Manager
        >>> manager = Manager()
        >>> queue = manager.Queue()
        >>> config = ProgressConfig(total_runs=30, total_folds=150, total_epochs=150000)
        >>> listener = ProgressListener(queue, config)
        >>> listener.start()
        >>> # ... workers send updates via queue.put(("epoch", 100)) ...
        >>> listener.join()  # Wait for natural completion
        >>> # Or: listener.stop()  # Force stop if needed
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

        Expected message format: (msg_type, count)
        - ("epoch", n): Update epoch bar by n
        - ("fold", n): Update fold bar by n
        - ("run", n): Update run bar by n
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

        finished_runs = 0

        try:
            while finished_runs < self.config.total_runs:
                # Check if stop was requested
                if self._stop_event.is_set():
                    break

                try:
                    msg, count = self.queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
                except queue.Empty:
                    # Timeout - loop back to check stop_event and continue waiting
                    continue

                # Check for shutdown sentinel
                if msg == _SHUTDOWN_SENTINEL[0]:
                    break

                if msg == "epoch":
                    self._pbar_epoch.update(count)
                elif msg == "fold":
                    self._pbar_fold.update(count)
                elif msg == "run":
                    self._pbar_exp.update(count)
                    finished_runs += count
        finally:
            # Ensure bars are closed properly
            if self._pbar_epoch is not None:
                self._pbar_epoch.close()
            if self._pbar_fold is not None:
                self._pbar_fold.close()
            if self._pbar_exp is not None:
                self._pbar_exp.close()


def send_progress(
    progress_queue: Optional[Queue], msg_type: str, count: int = 1
) -> None:
    """
    Send a progress update to the listener queue.

    This is a safe helper that no-ops if the queue is None (sequential mode).

    Args:
        progress_queue: Multiprocessing queue or None for sequential mode.
        msg_type: Type of progress ("epoch", "fold", or "run").
        count: Number to increment by (default: 1).
    """
    if progress_queue is not None:
        progress_queue.put((msg_type, count))
