import doctest
import time
import unittest
from multiprocessing import Queue

import hgp_lib.benchmarkers.progress
from hgp_lib.benchmarkers.progress import (
    ProgressConfig,
    ProgressListener,
    send_progress,
    _SHUTDOWN_SENTINEL,
)


class TestProgressConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ProgressConfig(total_runs=10, total_folds=50, total_epochs=5000)
        self.assertTrue(cfg.show_run_progress)
        self.assertTrue(cfg.show_fold_progress)
        self.assertTrue(cfg.show_epoch_progress)

    def test_custom_flags(self):
        cfg = ProgressConfig(
            total_runs=1,
            total_folds=2,
            total_epochs=100,
            show_run_progress=False,
            show_fold_progress=False,
            show_epoch_progress=False,
        )
        self.assertFalse(cfg.show_run_progress)
        self.assertFalse(cfg.show_fold_progress)
        self.assertFalse(cfg.show_epoch_progress)


class TestSendProgress(unittest.TestCase):
    def test_none_queue_is_noop(self):
        # Should not raise
        send_progress(None, "epoch", 5)

    def test_sends_to_queue(self):
        q = Queue()
        send_progress(q, "fold", 3)
        msg, count = q.get(timeout=1)
        self.assertEqual(msg, "fold")
        self.assertEqual(count, 3)

    def test_default_count(self):
        q = Queue()
        send_progress(q, "run")
        msg, count = q.get(timeout=1)
        self.assertEqual(count, 1)

    def test_multiple_messages(self):
        q = Queue()
        send_progress(q, "epoch", 10)
        send_progress(q, "fold", 1)
        send_progress(q, "run", 1)
        msgs = [q.get(timeout=1) for _ in range(3)]
        self.assertEqual(msgs, [("epoch", 10), ("fold", 1), ("run", 1)])


class TestProgressListener(unittest.TestCase):
    def _make_config(self, runs=2, folds=4, epochs=20):
        return ProgressConfig(
            total_runs=runs,
            total_folds=folds,
            total_epochs=epochs,
            show_run_progress=False,
            show_fold_progress=False,
            show_epoch_progress=False,
        )

    def test_start_and_stop(self):
        q = Queue()
        listener = ProgressListener(q, self._make_config())
        listener.start()
        self.assertTrue(listener._thread.is_alive())
        listener.stop()
        self.assertFalse(listener._thread.is_alive())

    def test_natural_completion(self):
        """Listener should exit when all runs are reported."""
        q = Queue()
        cfg = self._make_config(runs=2)
        listener = ProgressListener(q, cfg)
        listener.start()
        q.put(("run", 1))
        q.put(("run", 1))
        listener.join()
        self.assertFalse(listener._thread.is_alive())

    def test_epoch_and_fold_updates(self):
        q = Queue()
        cfg = self._make_config(runs=1, folds=2, epochs=10)
        listener = ProgressListener(q, cfg)
        listener.start()
        for _ in range(10):
            q.put(("epoch", 1))
        q.put(("fold", 1))
        q.put(("fold", 1))
        q.put(("run", 1))
        listener.join()
        self.assertFalse(listener._thread.is_alive())

    def test_stop_while_waiting(self):
        """stop() should unblock a listener waiting on an empty queue."""
        q = Queue()
        cfg = self._make_config(runs=100)
        listener = ProgressListener(q, cfg)
        listener.start()
        time.sleep(0.1)
        listener.stop()
        self.assertFalse(listener._thread.is_alive())

    def test_shutdown_sentinel(self):
        q = Queue()
        cfg = self._make_config(runs=100)
        listener = ProgressListener(q, cfg)
        listener.start()
        q.put(_SHUTDOWN_SENTINEL)
        listener._thread.join(timeout=10)
        self.assertFalse(listener._thread.is_alive())

    def test_unknown_message_type_ignored(self):
        """Unknown message types should not crash the listener."""
        q = Queue()
        cfg = self._make_config(runs=1)
        listener = ProgressListener(q, cfg)
        listener.start()
        q.put(("unknown_type", 99))
        q.put(("also_unknown", 1))
        q.put(("run", 1))  # complete normally
        listener.join()
        self.assertFalse(listener._thread.is_alive())

    def test_batch_run_count(self):
        """A single ("run", N) message with N > 1 should count all runs."""
        q = Queue()
        cfg = self._make_config(runs=3)
        listener = ProgressListener(q, cfg)
        listener.start()
        q.put(("run", 3))
        listener.join()
        self.assertFalse(listener._thread.is_alive())

    def test_join_without_start_is_noop(self):
        """join() on a never-started listener should not hang."""
        q = Queue()
        listener = ProgressListener(q, self._make_config())
        listener.join()  # _thread is None, should return immediately

    def test_stop_without_start_is_noop(self):
        """stop() on a never-started listener should not hang."""
        q = Queue()
        listener = ProgressListener(q, self._make_config())
        listener.stop()  # _thread is None, should return immediately

    def test_doctests(self):
        result = doctest.testmod(hgp_lib.benchmarkers.progress, verbose=False)
        self.assertEqual(result.failed, 0, f"Doctests failed: {result}")


if __name__ == "__main__":
    unittest.main()
