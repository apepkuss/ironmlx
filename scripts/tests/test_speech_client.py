"""Wire failure tests for the standalone macOS speech client.

Set IRONMLX_SPEECH_CLIENT to the compiled example. Set
IRONMLX_SPEECH_PLAYBACK_TESTS=1 to include real audio-device playback tests.
No model weights required; real model acceptance is a separate gate.
"""
import http.server
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave

CLIENT = os.environ.get("IRONMLX_SPEECH_CLIENT")
PLAYBACK = os.environ.get("IRONMLX_SPEECH_PLAYBACK_TESTS") == "1"
PCM = b"".join(struct.pack("<h", int(1200 * math.sin(i * 2 * math.pi * 440 / 22050))) for i in range(4410))


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)
        f.writeframes(PCM)
    return output.getvalue()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Expected when the client rejects headers or cancels a request.
            pass

    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        mode = self.path.strip("/")
        if mode == "json-error":
            data = b'{"error":{"message":"invalid reference"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if mode == "cancel-before-headers":
            time.sleep(0.5)
            self.close_connection = True
            return
        self.send_response(200)
        is_wav = request["response_format"] == "wav"
        self.send_header("Content-Type", "audio/wav" if is_wav else "audio/pcm")
        self.send_header("X-Audio-Sample-Rate", "24000" if mode == "wrong-rate" else "22050")
        self.send_header("X-Audio-Channels", "1")
        self.send_header("X-Audio-Sample-Format", "s16le")
        data = wav_bytes() if is_wav else PCM
        if is_wav:
            if mode == "bad-wav":
                data = b"NOPE" + data[4:]
            self.send_header("Content-Length", str(len(data)))
        else:
            self.send_header("X-IronMLX-Streaming-Granularity", "segment")
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if is_wav:
            self.wfile.write(data)
            return
        if mode == "odd-eof":
            data += b"\x01"
        try:
            for offset in range(0, len(data), 333):
                block = data[offset:offset + 333]
                self.wfile.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                self.wfile.flush()
                if mode == "cancel-stream":
                    time.sleep(0.06)
            # Device playback must occur before clean transport completion.
            if mode == "pcm":
                time.sleep(0.5)
            if mode == "broken-stream":
                self.close_connection = True
                return
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@unittest.skipUnless(CLIENT, "compile example and set IRONMLX_SPEECH_CLIENT")
class SpeechClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.reference = self.root / "reference.wav"
        self.reference.write_bytes(wav_bytes())
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def invoke(self, mode, fmt="pcm", extra=(), voice=None):
        self.output = self.root / f"{mode}.{fmt}"
        selector = ["--voice", voice] if voice else ["--reference", str(self.reference)]
        result = subprocess.run([
            CLIENT, "--url", f"http://127.0.0.1:{self.server.server_port}/{mode}",
            *selector, "--text", "client test",
            "--format", fmt, "--output", str(self.output), *extra,
        ], capture_output=True, text=True, timeout=15)
        self.assertFalse(list(self.root.glob("*.partial")))
        return result

    def assertFailure(self, mode, message, fmt="pcm", extra=()):
        result = self.invoke(mode, fmt, extra)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn('"status":"played"', result.stdout)
        self.assertFalse(self.output.exists())

    def test_json_failure_is_not_played(self):
        self.assertFailure("json-error", "HTTP 400")

    def test_wrong_rate_is_not_guessed(self):
        self.assertFailure("wrong-rate", "Unexpected X-Audio-Sample-Rate")

    def test_cancel_before_headers(self):
        self.assertFailure("cancel-before-headers", "speech client failed", extra=("--cancel-after-ms", "100"))

    def test_voice_selector_is_sent_without_reference_audio(self):
        result = self.invoke("json-error", voice="narrator")
        self.assertNotEqual(result.returncode, 0)
        request = self.server.requests[0]
        self.assertEqual(request["voice"], "narrator")
        self.assertNotIn("ref_audio", request)

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_pcm_odd_reads_play_before_eof(self):
        result = self.invoke("pcm")
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual(stats["played_frames"], len(PCM) // 2)
        self.assertLess(stats["first_played_ms"], stats["download_complete_ms"])
        self.assertLessEqual(stats["peak_queued_frames"], 44100)
        self.assertEqual(self.output.read_bytes(), PCM)
        self.assertTrue(self.server.requests[0]["stream"])

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_complete_wav_playback(self):
        result = self.invoke("wav", "wav")
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual(stats["played_frames"], len(PCM) // 2)
        self.assertGreater(stats["first_played_ms"], stats["download_complete_ms"])
        self.assertEqual(self.output.read_bytes(), wav_bytes())

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_odd_final_byte_fails(self):
        self.assertFailure("odd-eof", "unmatched final byte")

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_abnormal_transport_is_failure(self):
        self.assertFailure("broken-stream", "speech client failed")

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_cancel_active_stream(self):
        self.assertFailure("cancel-stream", "speech client failed", extra=("--cancel-after-ms", "250"))

    @unittest.skipUnless(PLAYBACK, "requires actual audio device")
    def test_invalid_wav_is_not_successful(self):
        self.assertFailure("bad-wav", "Invalid WAV", "wav")


if __name__ == "__main__":
    unittest.main()
