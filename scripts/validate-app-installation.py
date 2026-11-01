#!/usr/bin/env python3
"""Validate full-App installation, manual replacement upgrades and recovery on macOS.

Uses CFFIXED_USER_HOME for the App's Foundation stores and a private model copy.
Does not change HOME, install in /Applications, sign with production credentials,
or publish. This is local functional acceptance, not final release identity or
Sparkle transport acceptance. Output retains logs and hashes, not model weights.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
import urllib.request
import wave


REPO = Path(__file__).resolve().parent.parent


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        item = {"mode": stat.S_IMODE(path.lstat().st_mode)}
        if path.is_symlink():
            item.update(kind="symlink", target=os.readlink(path))
        elif path.is_file():
            item.update(kind="file", size=path.stat().st_size, sha256=sha(path))
        else:
            item.update(kind="directory")
        result[str(path.relative_to(root))] = item
    return result


def info(app):
    return plistlib.loads((app / "Contents/Info.plist").read_bytes())


def wait_for(predicate, timeout=150):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError) as error:
            last_error = error
        time.sleep(0.25)
    raise TimeoutError(f"Acceptance condition timed out: {last_error}")


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60 if body else 5) as response:
        return json.load(response)


def processes(app):
    result = {}
    prefix = str(app.resolve()) + "/Contents/"
    for line in subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and str(Path(fields[1]).resolve()).startswith(prefix):
            result[Path(fields[1]).name] = int(fields[0])
    return result


def preserved_files(root):
    paths = [root / "config/app_config.json", root / "model_params.json"]
    for directory in ("models", "voices", "acceptance-user-data"):
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths) if path.is_file()}


def infer(port, model):
    result = request(port, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Reply with exactly 42."}],
        "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    choice = result["choices"][0]
    assert choice["message"]["content"].strip(), result
    assert choice["finish_reason"] in ("stop", "length"), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", action="append", default=[], help="TAG=/absolute/App.app")
    parser.add_argument("--model", required=True, type=Path, help="Existing managed snapshot with manifest")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    assert not output.exists(), "Evidence output must not exist"
    model_manifest = json.loads((model / ".ironmlx-snapshot.json").read_text())
    model_id = model_manifest["repo_id"]
    baselines = [(tag, Path(path).resolve()) for tag, path in (value.split("=", 1) for value in args.baseline)]
    candidate_info = info(candidate)
    candidate_build = candidate_info["CFBundleVersion"]
    candidate_version = candidate_info["CFBundleShortVersionString"]
    assert candidate_build.isdecimal() and int(candidate_build) > 0
    for _, baseline in baselines:
        assert int(info(baseline)["CFBundleVersion"]) < int(candidate_build), "Upgrade build must increase"
    assert not any("/IronMLX.app/Contents/" in line for line in subprocess.check_output(
        ["ps", "-axo", "comm="], text=True).splitlines()), "Quit existing IronMLX before acceptance"
    output.mkdir(parents=True)
    production_paths = [Path.home() / ".ironmlx/config/app_config.json", Path.home() / ".ironmlx/model_params.json"]
    production_before = {str(path): sha(path) if path.exists() else None for path in production_paths}
    summary = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate_info": candidate_info, "scope": "full App; manual ZIP/DMG installation and replacement upgrades",
        "model": {"source": str(model), "manifest": model_manifest}, "cases": {},
        "limitations": ["local ad-hoc candidate from working tree", "not final stable tag/signing/notarization acceptance",
                        "manual replacement upgrades; does not exercise production Sparkle transport"],
    }
    dump(output / "candidate-manifest.json", manifest(candidate))
    process = None
    log_handle = None
    current_app = None
    with tempfile.TemporaryDirectory(prefix="ironmlx-install-acceptance-") as temporary:
        work = Path(temporary)
        control_source = work / "quit.swift"
        control_source.write_text("import AppKit\nimport Foundation\n"
                                  "let pid = pid_t(CommandLine.arguments[1])!\n"
                                  "guard let app = NSRunningApplication(processIdentifier: pid), app.terminate() "
                                  "else { exit(1) }\n")
        control = work / "quit-app"
        run("swiftc", control_source, "-o", control)

        def stop():
            nonlocal process, log_handle
            if process and process.poll() is None:
                run(control, process.pid)
                process.wait(timeout=30)
                assert process.returncode == 0, process.returncode
            if current_app:
                wait_for(lambda: not processes(current_app), timeout=30)
            if log_handle:
                log_handle.close()
            process = None
            log_handle = None

        def launch(app, home, label, port):
            nonlocal process, log_handle, current_app
            current_app = app
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} is already occupied"
            environment = dict(os.environ, CFFIXED_USER_HOME=str(home))
            log_handle = (output / f"{label}.log").open("w")
            process = subprocess.Popen([str(app / "Contents/MacOS/IronMLX")], env=environment,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            wait_for(lambda: request(port, "/healthz").get("status") == "healthy")
            assert process.poll() is None
            assert (home / ".ironmlx/logs").exists(), "App did not use isolated Foundation home"
            return request(port, "/healthz")

        def loaded(port):
            models = request(port, "/admin/api/models/loaded")
            return any(item["id"] == model_id and item.get("pinned") for item in models)

        try:
            # Archive creation verifies actual ZIP extraction and read-only DMG mounting.
            archive = output / "archives"
            with (output / "archive.log").open("w") as log:
                run("python3", REPO / "scripts/release-archives.py", "assemble", candidate, archive,
                    stdout=log, stderr=subprocess.STDOUT)
            for kind, source in [("zip", archive / f"IronMLX-{candidate_version}.zip"), ("dmg", archive / "IronMLX.dmg")]:
                case = work / ("fresh-" + kind)
                app = case / "Applications/IronMLX.app"
                home = case / "home"
                home.mkdir(parents=True)
                if kind == "zip":
                    run("ditto", "-x", "-k", source, case / "unpacked")
                    run("ditto", case / f"unpacked/IronMLX-{candidate_version}/IronMLX.app", app)
                else:
                    mount = case / "mounted"
                    mount.mkdir()
                    run("hdiutil", "attach", source, "-readonly", "-nobrowse", "-mountpoint", mount, "-quiet")
                    try:
                        run("ditto", mount / "IronMLX.app", app)
                    finally:
                        run("hdiutil", "detach", mount, "-quiet")
                assert manifest(app) == manifest(candidate)
                run("codesign", "--verify", "--deep", "--strict", app)
                # Fresh installs use the product's default port, without seeded configuration.
                health = launch(app, home, "fresh-" + kind, 9068)
                assert request(9068, "/v1/models")["data"] == []
                stop()
                summary["cases"]["fresh-" + kind] = {"passed": True, "health": health,
                    "bundle_manifest_matches": True, "clean_shutdown": True}
                dump(output / "results.json", summary)

            for index, (tag, baseline) in enumerate(baselines):
                label = tag.replace(".", "-")
                case = work / label
                app = case / "Applications/IronMLX.app"
                home = case / "home"
                root = home / ".ironmlx"
                root.mkdir(parents=True)
                repository = root / "models/huggingface" / model_id.replace("/", "--")
                snapshot = repository / "snapshots" / model_manifest["commit_sha"]
                snapshot.mkdir(parents=True)
                # Dereference legacy cache links and clone regular files on APFS.
                for path in model.rglob("*"):
                    if path.is_file():
                        target = snapshot / path.relative_to(model)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        run("cp", "-c", path.resolve(), target)
                (repository / "refs").mkdir()
                (repository / "refs/main").write_text(model_manifest["commit_sha"] + "\n")
                for item in model_manifest["files"]:
                    path = snapshot / item["path"]
                    assert path.stat().st_size == item["size"] and sha(path) == item["sha256"], item["path"]
                port = 19780 + index
                config = root / "config/app_config.json"
                config.parent.mkdir()
                dump(config, {"schema_version": 1, "payload": {
                    "host": "127.0.0.1", "port": port, "network_mode": "local",
                    "default_model": model_id, "loaded_models": [model_id], "pinned_models": [model_id],
                    "language": "en", "theme": "dark", "log_level": "INFO", "cache_enable": False,
                    "max_models": 2, "verify_model_on_load": False,
                }})
                dump(root / "model_params.json", {"schema_version": 1, "models": {model_id: {
                    "model_id": model_id, "alias": "Release acceptance", "context_size": "4096",
                    "max_tokens": "4096", "temperature": "0.1", "top_p": "0.9",
                }}})
                (root / "voices").mkdir()
                with wave.open(str(root / "voices/reference.wav"), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(bytes(3200))
                (root / "acceptance-user-data").mkdir()
                (root / "acceptance-user-data/sentinel.txt").write_text("Preserve user data during App replacement.\n")
                run("ditto", baseline, app)
                baseline_health = launch(app, home, label + "-before", port)
                wait_for(lambda: loaded(port))
                before_inference = infer(port, model_id)
                stop()
                before = preserved_files(root)
                dump(output / (label + "-data-before.json"), before)
                baseline_info = info(app)
                shutil.rmtree(app)
                run("ditto", candidate, app)
                assert preserved_files(root) == before, "App replacement changed external data"
                health = launch(app, home, label + "-after", port)
                wait_for(lambda: loaded(port))
                after_inference = infer(port, model_id)
                assert preserved_files(root) == before, "Startup changed preserved data"
                before_processes = processes(app)
                assert set(before_processes) == {"IronMLX", "ironmlx"}, before_processes
                os.kill(before_processes["ironmlx"], signal.SIGKILL)
                wait_for(lambda: processes(app).get("ironmlx") not in (None, before_processes["ironmlx"]) and loaded(port))
                after_processes = processes(app)
                assert before_processes["IronMLX"] == after_processes["IronMLX"]
                recovery_inference = infer(port, model_id)
                assert preserved_files(root) == before, "Crash recovery changed preserved data"
                stop()
                launch(app, home, label + "-restart", port)
                wait_for(lambda: loaded(port))
                restart_inference = infer(port, model_id)
                stop()
                after = preserved_files(root)
                assert after == before, "Restart changed preserved data"
                dump(output / (label + "-data-after.json"), after)
                assert manifest(app) == manifest(candidate)
                for log in (root / "logs").glob("*"):
                    if log.is_file():
                        shutil.copy2(log, output / (label + "-" + log.name))
                summary["cases"][tag] = {
                    "passed": True, "baseline_info": baseline_info, "baseline_health": baseline_health,
                    "installed_build": info(app)["CFBundleVersion"], "health": health,
                    "before_inference": before_inference, "after_inference": after_inference,
                    "recovery_inference": recovery_inference, "restart_inference": restart_inference,
                    "processes_before_crash": before_processes, "processes_after_recovery": after_processes,
                    "pinned_model_restored": True, "preserved_file_count": len(before),
                    "all_preserved_hashes_match": True, "bundle_manifest_matches": True, "clean_shutdown": True,
                }
                dump(output / "results.json", summary)
                print(f"Passed full-App replacement {tag} -> build {candidate_build}, model restore, crash recovery and data preservation", flush=True)
        finally:
            try:
                stop()
            finally:
                if current_app:
                    for pid in processes(current_app).values():
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                production_after = {str(path): sha(path) if path.exists() else None for path in production_paths}
                summary["production_configuration_unchanged"] = production_before == production_after
                dump(output / "results.json", summary)
                assert production_before == production_after, "Production configuration changed"
    summary["temporary_installations_and_model_copies_removed"] = True
    summary["passed"] = True
    summary["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dump(output / "results.json", summary)
    print("Full-App installation acceptance passed; temporary installations and model copies removed", flush=True)


if __name__ == "__main__":
    main()
