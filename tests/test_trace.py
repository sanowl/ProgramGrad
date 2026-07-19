import json
import tempfile
import threading
import unittest
from pathlib import Path

from programgrad import Tensor, soft_if, soft_select, trace, training_mode, training_trace
from programgrad.trace import hard_shadow_enabled, ops_recording_enabled


class TraceTests(unittest.TestCase):
    def test_trace_configuration_is_validated(self):
        with self.assertRaisesRegex(TypeError, "mode must be a string"):
            trace(mode=1)
        with self.assertRaisesRegex(ValueError, "relaxation must not be empty"):
            trace(relaxation="  ")
        with self.assertRaisesRegex(TypeError, "fidelity must be a bool"):
            trace(fidelity="yes")
        with self.assertRaisesRegex(TypeError, "record_ops must be a bool"):
            trace(record_ops=1)
        with self.assertRaisesRegex(TypeError, "hard_shadow must be a bool"):
            trace(hard_shadow=1)
        with self.assertRaisesRegex(TypeError, "hard_shadow must be a bool"):
            with training_mode(hard_shadow=1):
                pass

    def test_hard_path_preserves_mixed_event_order(self):
        with trace(record_ops=False) as tr:
            soft_select([1.0, 2.0], [0.0, 1.0])
            soft_if(1.0, 3.0, 0.0)

        self.assertTrue(tr.hard_path()[0].startswith("search#"))
        self.assertTrue(tr.hard_path()[1].startswith("branch#"))

    def test_trace_stack_is_isolated_between_threads(self):
        main_trace = trace(record_ops=True)
        worker_trace = trace(record_ops=True)
        worker_entered = threading.Event()
        release_worker = threading.Event()

        def worker() -> None:
            with worker_trace:
                worker_entered.set()
                release_worker.wait(timeout=2.0)
                Tensor(1.0) + Tensor(2.0)

        thread = threading.Thread(target=worker)
        with main_trace:
            thread.start()
            self.assertTrue(worker_entered.wait(timeout=2.0))
            try:
                Tensor(3.0) + Tensor(4.0)
            finally:
                release_worker.set()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(main_trace.ops), 1)
        self.assertEqual(len(worker_trace.ops), 1)

    def test_exports_create_parent_directories(self):
        with trace(record_ops=False) as tr:
            soft_if(1.0, 2.0, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = tr.export_json(root / "json" / "trace.json")
            svg_path = tr.export_svg(root / "svg" / "trace.svg")

            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["mode"], "dual")
            self.assertTrue(payload["fidelity"])
            self.assertFalse(payload["record_ops"])
            self.assertTrue(payload["hard_shadow"])
            self.assertIn("<svg", svg_path.read_text())

    def test_default_trace_does_not_record_ops(self):
        with trace() as tr:
            Tensor(1.0, requires_grad=True) + Tensor(2.0)
            self.assertFalse(ops_recording_enabled())
        self.assertEqual(tr.ops, [])

    def test_training_mode_disables_hard_shadow(self):
        value = Tensor(1.0)
        value.hard_value = 9.0
        self.assertTrue(hard_shadow_enabled())
        with training_mode(hard_shadow=False):
            self.assertFalse(hard_shadow_enabled())
            self.assertFalse(ops_recording_enabled())
            out = value * 2.0
            self.assertIsNone(out.hard_value)
            self.assertAlmostEqual(out.data, 2.0)
        self.assertTrue(hard_shadow_enabled())
        restored = value * 2.0
        self.assertAlmostEqual(restored.hard_value, 18.0)

    def test_training_trace_defaults_are_cheap(self):
        with training_trace() as tr:
            soft_if(1.0, 2.0, 0.0)
            self.assertFalse(ops_recording_enabled())
            self.assertTrue(hard_shadow_enabled())
        self.assertFalse(tr.fidelity)
        self.assertFalse(tr.record_ops_enabled)
        self.assertEqual(tr.ops, [])
        self.assertEqual(len(tr.branches), 1)
        self.assertIsNone(tr.branches[0].fidelity)


if __name__ == "__main__":
    unittest.main()
