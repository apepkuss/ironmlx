import contextlib
import errno
import importlib.util
import io
from pathlib import Path
import socket
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "backend_crash_helper",
    ROOT / "ironmlx-app/Tests/IronMLXAppCoreTests/Fixtures/backend_crash_helper.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class CrashHelperStartupTests(unittest.TestCase):
    def test_loopback_bind_does_not_use_dns(self):
        with mock.patch("socket.getfqdn", side_effect=AssertionError("DNS must not be used")):
            with helper.bind_server("127.0.0.1", 0, 0) as server:
                self.assertEqual(server.server_name, "127.0.0.1")
                self.assertGreater(server.server_port, 0)

    def test_contended_port_is_bound_after_handoff(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with mock.patch.object(helper.time, "sleep", side_effect=lambda _: reservation.close()) as sleep:
                    with helper.bind_server("127.0.0.1", port, 1) as server:
                        self.assertEqual(server.server_port, port)
                sleep.assert_called_once_with(.05)
            self.assertIn("helper waiting for port handoff", output.getvalue())

    def test_port_contention_has_a_finite_deadline(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(helper.time, "monotonic", side_effect=[0, 0, 1]):
                    with mock.patch.object(helper.time, "sleep") as sleep:
                        with self.assertRaises(OSError) as caught:
                            helper.bind_server("127.0.0.1", port, .5)
            self.assertEqual(caught.exception.errno, errno.EADDRINUSE)
            sleep.assert_called_once_with(.05)

    def test_other_bind_errors_are_not_retried(self):
        with mock.patch.object(helper, "LoopbackHTTPServer", side_effect=OSError(errno.EACCES, "denied")):
            with mock.patch.object(helper.time, "sleep") as sleep:
                with self.assertRaises(OSError) as caught:
                    helper.bind_server("127.0.0.1", 0, 1)
        self.assertEqual(caught.exception.errno, errno.EACCES)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
