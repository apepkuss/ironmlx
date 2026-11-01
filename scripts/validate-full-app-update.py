#!/usr/bin/env python3
"""Exercise full IronMLX Apps through Sparkle on isolated loopback HTTPS.

Uses test-only Bundle derivatives and an ephemeral Ed25519 key. Baseline and
candidate code must remain unchanged. No production feed or /Applications
installation is modified. Owned TLS trust is removed on exit; caller-supplied
TLS credentials and trust remain the caller's cleanup responsibility.
"""
import argparse
import datetime
import functools
import http.server
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave


REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("installation", Path(__file__).with_name("validate-app-installation.py"))
installation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installation)
run, dump, sha = installation.run, installation.dump, installation.sha
info, manifest, wait_for = installation.info, installation.manifest, installation.wait_for
request, processes = installation.request, installation.processes
preserved_files, infer = installation.preserved_files, installation.infer


def code_sha(path):
    """Hash an arm64 Mach-O excluding only its code-signature load command/blob."""
    import hashlib
    data = bytearray(path.read_bytes())
    assert struct.unpack_from("<I", data)[0] == 0xFEEDFACF
    offset = 32
    for _ in range(struct.unpack_from("<I", data, 16)[0]):
        command, size = struct.unpack_from("<II", data, offset)
        if command == 0x19 and bytes(data[offset + 8:offset + 24]).rstrip(b"\0") == b"__LINKEDIT":
            # Re-signing changes the signature blob length and this segment's
            # file/virtual sizes, including when switching Developer ID/ad-hoc.
            struct.pack_into("<Q", data, offset + 32, 0)
            struct.pack_into("<Q", data, offset + 48, 0)
        if command == 0x1D:
            signature_offset, signature_size = struct.unpack_from("<II", data, offset + 8)
            assert signature_offset + signature_size == len(data)
            return hashlib.sha256(data[:offset] + bytes(size) + data[offset + size:signature_offset]).hexdigest()
        offset += size
    raise ValueError("Expected a signed arm64 executable")


def seed(home, model, model_info, port):
    root = home / ".ironmlx"
    repo = root / "models/huggingface" / model_info["repo_id"].replace("/", "--")
    snapshot = repo / "snapshots" / model_info["commit_sha"]
    snapshot.mkdir(parents=True)
    for path in model.rglob("*"):
        if path.is_file():
            target = snapshot / path.relative_to(model)
            target.parent.mkdir(parents=True, exist_ok=True)
            run("cp", "-c", path.resolve(), target)
    for item in model_info["files"]:
        path = snapshot / item["path"]
        assert path.stat().st_size == item["size"] and sha(path) == item["sha256"]
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(model_info["commit_sha"] + "\n")
    config = root / "config/app_config.json"
    config.parent.mkdir()
    model_id = model_info["repo_id"]
    dump(config, {"schema_version": 1, "payload": {
        "host": "127.0.0.1", "port": port, "network_mode": "local", "language": "en",
        "theme": "dark", "log_level": "INFO", "cache_enable": False, "max_models": 2,
        "default_model": model_id, "loaded_models": [model_id], "pinned_models": [model_id],
        "verify_model_on_load": False,
    }})
    dump(root / "model_params.json", {"schema_version": 1, "models": {model_id: {
        "model_id": model_id, "alias": "Sparkle acceptance", "context_size": "4096",
        "max_tokens": "4096", "temperature": "0.1", "top_p": "0.9",
    }}})
    (root / "voices").mkdir()
    with wave.open(str(root / "voices/reference.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(3200))
    (root / "acceptance-user-data").mkdir()
    (root / "acceptance-user-data/sentinel.txt").write_text("Preserve external data through Sparkle.\n")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, action="append", help="TAG=/absolute/App.app")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--signing-identity", required=True, help="Matching Developer ID identity for test derivatives only")
    parser.add_argument("--tls-certificate", type=Path, help="Caller-managed, already-trusted loopback certificate")
    parser.add_argument("--tls-key", type=Path, help="Caller-managed TLS private key; requires --tls-certificate")
    args = parser.parse_args()
    assert bool(args.tls_certificate) == bool(args.tls_key), "Supply both TLS paths or neither"
    managed_tls = args.tls_certificate is None
    candidate, model, output = args.candidate.resolve(), args.model.resolve(), args.output.resolve()
    baselines = [(tag, Path(path).resolve()) for tag, path in (value.split("=", 1) for value in args.baseline)]
    assert not output.exists(), "Output must not exist"
    assert not any("/IronMLX.app/Contents/" in line for line in subprocess.check_output(
        ["ps", "-axo", "comm="], text=True).splitlines()), "Quit existing IronMLX"
    target_info = info(candidate)
    identity_match = re.search(r"\(([A-Z0-9]+)\)$", args.signing_identity)
    assert identity_match is not None, "A Developer ID identity with a Team ID is required"
    for original in [candidate, *[app for _, app in baselines]]:
        for executable in [original, original / "Contents/Frameworks/Sparkle.framework/Versions/B/Autoupdate"]:
            details = subprocess.run(["codesign", "-d", "--verbose=2", str(executable)], text=True,
                                     capture_output=True, check=True).stderr
            assert "TeamIdentifier=" + identity_match[1] in details, "All Apps and Sparkle helpers must share the signing team"
    for _, baseline in baselines:
        assert int(info(baseline)["CFBundleVersion"]) < int(target_info["CFBundleVersion"])
    model_info = json.loads((model / ".ironmlx-snapshot.json").read_text())
    model_id = model_info["repo_id"]
    bins = REPO / "ironmlx-app/.build/artifacts/sparkle/Sparkle/bin"
    assert (bins / "generate_appcast").is_file()
    production_paths = [Path.home() / ".ironmlx/config/app_config.json", Path.home() / ".ironmlx/model_params.json",
                        Path.home() / "Library/Preferences/com.ironmlx.app.plist"]
    original_data = {str(p): sha(p) if p.exists() else None for p in production_paths}
    output.mkdir(parents=True)
    original_bundle = manifest(candidate)
    dump(output / "original-candidate-manifest.json", original_bundle)
    result = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "candidate_info": target_info, "model_source": str(model), "cases": {},
              "scope": "full App Sparkle; loopback HTTPS, ephemeral key, test-only metadata/signature derivatives",
              "tls_lifecycle": "tool-managed" if managed_tls else "caller-managed",
              "production_feed_modified": False}
    current_app, server, thread, certificate, fingerprint = None, None, None, None, None
    trusted = False
    keychain = Path.home() / "Library/Keychains/login.keychain-db"
    cleanup = {}
    preference_domains = []
    with tempfile.TemporaryDirectory(prefix="ironmlx-full-update-") as temporary:
        work = Path(temporary)
        control_source = work / "quit.swift"
        control_source.write_text("import AppKit\nimport Foundation\n"
                                  "let pid = pid_t(CommandLine.arguments[1])!\n"
                                  "guard let app = NSRunningApplication(processIdentifier: pid), app.terminate() else { exit(1) }\n")
        control = work / "quit-app"
        run("swiftc", control_source, "-o", control)

        def stop():
            if current_app:
                pid = processes(current_app).get("IronMLX")
                if pid:
                    run(control, pid)
                wait_for(lambda: not processes(current_app), timeout=30)

        try:
            private = work / "update.key"
            public = subprocess.check_output(["swift", str(REPO / "scripts/generate-development-update-key.swift"),
                                              str(private)], text=True).strip()
            if managed_tls:
                certificate, tls_key = work / "localhost.pem", work / "localhost.key"
                run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", "1",
                    "-keyout", tls_key, "-out", certificate, "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1", "-addext", "extendedKeyUsage=serverAuth",
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.chmod(tls_key, 0o600)
            else:
                certificate, tls_key = args.tls_certificate.resolve(), args.tls_key.resolve()
            fingerprint = subprocess.check_output(["openssl", "x509", "-in", str(certificate), "-noout",
                                                  "-fingerprint", "-sha1"], text=True).strip().split("=")[1].replace(":", "")
            if managed_tls:
                run("security", "add-trusted-cert", "-r", "trustRoot", "-p", "ssl", "-s", "127.0.0.1",
                    "-k", keychain, certificate, timeout=300)
                trusted = True
            for index, (tag, baseline) in enumerate(baselines):
                modes = ["bad-signature", "interrupted", "success"] if index == 0 else ["success"]
                for mode_index, mode in enumerate(modes):
                    label = tag + "-" + mode
                    case = work / label
                    # CFPreferences is not isolated by CFFIXED_USER_HOME when
                    # cfprefsd is shared. Give both derivatives a private domain.
                    domain = "com.ironmlx.full-update-validation." + uuid.uuid4().hex
                    preference_domains.append(domain)
                    home = case / "home"
                    port = 19820 + index * 4 + mode_index
                    root = seed(home, model, model_info, port)
                    feed = case / "feed"
                    feed.mkdir()
                    release = threading.Event()
                    events = []

                    class Handler(http.server.SimpleHTTPRequestHandler):
                        def log_message(self, *_):
                            pass

                        def do_GET(self):
                            events.append({"path": self.path, "time": time.time()})
                            if self.path == "/appcast.xml":
                                if not release.wait(timeout=180):
                                    self.send_error(503)
                                    return
                            if self.path == "/update.zip" and mode != "success":
                                data = (feed / "update.zip").read_bytes()
                                self.send_response(200)
                                self.send_header("Content-Length", str(len(data)))
                                self.end_headers()
                                if mode == "bad-signature":
                                    data = data[:-1] + bytes([data[-1] ^ 1])
                                    self.wfile.write(data)
                                    events.append({"injection": "tampered-archive", "bytes": len(data)})
                                else:
                                    self.wfile.write(data[:1024 * 1024])
                                    self.wfile.flush()
                                    events.append({"injection": "interrupted-download", "sent": 1024 * 1024,
                                                   "expected": len(data)})
                                    self.connection.shutdown(socket.SHUT_RDWR)
                                    self.connection.close()
                                return
                            return super().do_GET()

                    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(feed)))
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(certificate, tls_key)
                    server.socket = context.wrap_socket(server.socket, server_side=True)
                    feed_url = f"https://127.0.0.1:{server.server_port}/appcast.xml"
                    payloads = {}
                    for name, original in [("source", baseline), ("target", candidate)]:
                        app = case / name / "IronMLX.app"
                        run("ditto", original, app)
                        run("python3", REPO / "scripts/configure-app-updates.py", "development", feed_url,
                            public, "--plist", app / "Contents/Info.plist")
                        data = info(app)
                        data["CFBundleIdentifier"] = domain
                        data["LSEnvironment"] = {"CFFIXED_USER_HOME": str(home)}
                        (app / "Contents/Info.plist").write_bytes(plistlib.dumps(data))
                        command = ["codesign", "--force", "--sign", args.signing_identity]
                        if args.signing_identity != "-":
                            command += ["--options", "runtime", "--timestamp"]
                        run(*command, app, stderr=subprocess.DEVNULL)
                        run("codesign", "--verify", "--deep", "--strict", app)
                        payloads[name] = {"info": info(app), "original_info": info(original)}
                        original_rows, modified_rows = manifest(original), manifest(app)
                        excluded = {"Contents/Info.plist", "Contents/_CodeSignature/CodeResources", "Contents/MacOS/IronMLX"}
                        assert {k: v for k, v in original_rows.items() if k not in excluded} == {
                            k: v for k, v in modified_rows.items() if k not in excluded}, "Unexpected derivative changes"
                        assert code_sha(original / "Contents/MacOS/IronMLX") == code_sha(app / "Contents/MacOS/IronMLX")
                    archive = feed / "update.zip"
                    run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", case / "target/IronMLX.app", archive)
                    run(bins / "generate_appcast", "--ed-key-file", private, "--maximum-deltas", "0",
                        "--download-url-prefix", f"https://127.0.0.1:{server.server_port}/", feed,
                        stdout=subprocess.DEVNULL)
                    run(bins / "sign_update", "--verify", "--ed-key-file", private, feed / "appcast.xml",
                        stdout=subprocess.DEVNULL)
                    current_app = case / "install/IronMLX.app"
                    run("ditto", case / "source/IronMLX.app", current_app)
                    original_installed = manifest(current_app)
                    target_installed = manifest(case / "target/IronMLX.app")
                    marker = case / "update-ready"
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    run("open", "-n", current_app, "--args", "--ironmlx-development-update-test-marker", marker)

                    def loaded():
                        return any(item["id"] == model_id and item.get("pinned")
                                   for item in request(port, "/admin/api/models/loaded"))

                    wait_for(loaded)
                    before_pids = processes(current_app)
                    assert "IronMLX" in before_pids and "ironmlx" in before_pids
                    assert (root / "logs/app.log").exists(), "App ignored isolated Foundation home"
                    before_inference = infer(port, model_id)
                    before_data = preserved_files(root)
                    dump(output / (label + "-data-before.json"), before_data)
                    release.set()
                    if mode == "success":
                        def installed():
                            if marker.with_suffix(".error").exists():
                                raise RuntimeError(marker.with_suffix(".error").read_text())
                            return info(current_app)["CFBundleVersion"] == target_info["CFBundleVersion"]
                        wait_for(installed, timeout=180)
                        wait_for(lambda: processes(current_app).get("IronMLX") not in (None, before_pids["IronMLX"]) and loaded())
                        assert manifest(current_app) == target_installed
                        assert marker.is_file(), "Production updater did not reach its install callback"
                    else:
                        wait_for(lambda: marker.with_suffix(".error").is_file(), timeout=180)
                        assert manifest(current_app) == original_installed
                        assert processes(current_app)["IronMLX"] == before_pids["IronMLX"]
                        assert any("injection" in event for event in events), "Did not reach injected boundary"
                        error = marker.with_suffix(".error").read_text().lower()
                        assert "running the updater" not in error, "Updater helper failed before intended rejection"
                    after_pids = processes(current_app)
                    after_inference = infer(port, model_id)
                    assert preserved_files(root) == before_data, "Sparkle changed external data"
                    stop()
                    assert preserved_files(root) == before_data
                    for log in (root / "logs").glob("*"):
                        if log.is_file():
                            shutil.copy2(log, output / (label + "-" + log.name))
                    dump(output / (label + "-data-after.json"), preserved_files(root))
                    result["cases"][label] = {
                        "passed": True, "mode": mode, "before_processes": before_pids,
                        "after_processes": after_pids, "before_inference": before_inference,
                        "after_inference": after_inference, "derivatives": payloads,
                        "archive_sha256": sha(archive), "feed_sha256": sha(feed / "appcast.xml"),
                        "events": events, "preserved_files": len(before_data), "all_data_hashes_match": True,
                        "installed_build": info(current_app)["CFBundleVersion"],
                        "update_error": marker.with_suffix(".error").read_text() if mode != "success" else None,
                    }
                    dump(output / "results.json", result)
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=10)
                    server = None
                    print(f"Full App Sparkle passed: {tag}, {mode}, installed build {info(current_app)['CFBundleVersion']}", flush=True)
        except BaseException as error:
            result["failure"] = {"type": type(error).__name__, "message": str(error)}
            dump(output / "results.json", result)
            print("Full App Sparkle failed: " + repr(error), flush=True)
            raise
        finally:
            try:
                stop()
                cleanup["test_processes_stopped"] = True
            finally:
                if server:
                    if thread and thread.is_alive():
                        server.shutdown()
                    server.server_close()
                if trusted:
                    r = subprocess.run(["security", "remove-trusted-cert", str(certificate)], capture_output=True, timeout=300)
                    cleanup["trust_removed"] = r.returncode == 0
                if fingerprint and managed_tls:
                    subprocess.run(["security", "delete-certificate", "-Z", fingerprint, str(keychain)], capture_output=True)
                    certificates = subprocess.check_output(["security", "find-certificate", "-a", "-Z", str(keychain)], text=True)
                    cleanup["certificate_removed"] = fingerprint not in certificates.replace(":", "")
                for domain in preference_domains:
                    subprocess.run(["defaults", "delete", domain], capture_output=True)
                cleanup["original_candidate_unchanged"] = manifest(candidate) == original_bundle
                cleanup["production_configuration_unchanged"] = original_data == {
                    str(p): sha(p) if p.exists() else None for p in production_paths}
                result["cleanup"] = cleanup
                dump(output / "results.json", result)
                assert all(cleanup.values()), cleanup
    result["temporary_update_key_installations_and_model_copies_removed"] = True
    result["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["passed"] = True
    dump(output / "results.json", result)
    print("Full App Sparkle acceptance passed; temporary update key and installations removed", flush=True)


if __name__ == "__main__":
    main()
