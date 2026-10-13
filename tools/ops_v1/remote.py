#!/usr/bin/env python3
"""Access and deploy the independent v1 server with ignored private SSH config."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/autodl-tmp/other-inertial-benchmark-v1"
CPU_LOCK = "/root/autodl-tmp/inertial-benchmark/.local/campaigns/cpu-resource.lock"
GPU_LOCK = "/root/autodl-tmp/inertial-benchmark/.local/campaigns/gpu-0.lock"
EXCLUDED = (".git/", ".local/", "datasets/", "runs/", "external/", "data/raw/")


def connect(path: Path):
    import paramiko

    if path.stat().st_mode & 0o077:
        raise PermissionError("Private SSH configuration must have mode 0600")
    config = json.loads(path.read_text())
    if config["remote_root"] != REMOTE_ROOT:
        raise ValueError("This v1 tool requires the independent v1 remote root")
    hosts = path.parent / "autodl_known_hosts"
    client = paramiko.SSHClient()
    if hosts.exists():
        client.load_host_keys(str(hosts))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(config["host"], port=config["port"], username=config["user"],
                   password=config.get("password"), timeout=20, auth_timeout=20,
                   banner_timeout=20, look_for_keys=False, allow_agent=False)
    client.save_host_keys(str(hosts))
    hosts.chmod(0o600)
    client.get_transport().set_keepalive(30)
    return client


def execute(client, command: str, timeout: float) -> int:
    channel = client.get_transport().open_session(timeout=20)
    channel.exec_command(command)
    start = time.monotonic()
    try:
        while True:
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
                sys.stderr.buffer.flush()
            if (channel.exit_status_ready() and not channel.recv_ready()
                    and not channel.recv_stderr_ready()):
                return channel.recv_exit_status()
            if time.monotonic() - start > timeout:
                raise TimeoutError("SSH command timed out; use detached workers for long checks")
            time.sleep(0.05)
    finally:
        channel.close()


def deploy(client, selected: list | None, timeout: float) -> int:
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    available = {name for name in listed if name and not name.startswith(EXCLUDED)
                 and not (ROOT / name).is_symlink() and (ROOT / name).is_file()
                 and not any(part in ("__pycache__", ".pytest_cache", ".ruff_cache")
                             for part in Path(name).parts)
                 and not Path(name).name.startswith(".env")
                 and not name.endswith((".h5", ".pt", ".pth", ".npz", ".pyc"))}
    if selected and set(selected) - available:
        raise ValueError("Selected deployment paths must be admitted repository source files")
    names = sorted(selected or available)
    stream = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in names:
            value = (ROOT / name).read_bytes()
            info = archive.gettarinfo(str(ROOT / name), arcname=name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
            hashes[name] = hashlib.sha256(value).hexdigest()
    archive_bytes = stream.getvalue()
    timestamp = datetime.now(timezone.utc).isoformat()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    location = f"{REMOTE_ROOT}/.local/deployments/{archive_sha[:16]}"
    code = execute(client, "mkdir -p " + shlex.quote(location), timeout)
    if code:
        return code
    with client.open_sftp() as sftp:
        with sftp.file(REMOTE_ROOT + "/source_manifest.json", "r") as handle:
            previous_bytes = handle.read()
            previous = json.loads(previous_bytes)
        previous_sha = hashlib.sha256(previous_bytes).hexdigest()
        if selected:
            files = {**previous.get("files", {}),
                     **previous.get("operational_acceptance_helpers", {}),
                     **hashes}
        else:
            files = hashes
        stale = [] if selected else sorted(set(previous.get("files", {})) - set(files))
        identity = {"files": files, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()}
        manifest = {**identity, "schema": "ib.dev-v1.deployment.v1", "created_utc": timestamp,
                    "stage": "incremental" if selected else "integrated",
                    "git_worktree_dirty": bool(subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT)),
                    "archive_sha256": archive_sha, "deployed_paths": names,
                    "removed_previously_managed_source_files": stale,
                    "previous_source_manifest_sha256": previous_sha,
                    "deployment_id": hashlib.sha256(
                        json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                    "credentials_included": False, "production_source_modified": False}
        with sftp.file(location + "/source.tar.gz", "wb") as handle:
            handle.write(archive_bytes)
        with sftp.file(location + "/manifest.json", "w") as handle:
            handle.write(json.dumps(manifest, indent=2))
    pruning = ("import hashlib,json;from pathlib import Path;"
               f"root=Path({REMOTE_ROOT!r});stale=json.loads({json.dumps(stale)!r});"
               f"old=json.loads({json.dumps(previous.get('files', {}))!r});"
               f"excluded={EXCLUDED!r};\n"
               "if hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()"
               f"!={previous_sha!r}:"
               "raise RuntimeError('deployment manifest advanced; stage a fresh snapshot')\n"
               "for name in stale:\n"
               " p=root/name\n"
               " if name.startswith(excluded) or Path(name).is_absolute() "
               "or '..' in Path(name).parts "
               "or p.is_symlink():raise RuntimeError('unsafe managed removal path')\n"
               " if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=old[name]:"
               "raise RuntimeError('managed source changed before removal')\n"
               "for name in stale:\n"
               " p=root/name\n"
               " if p.exists():p.unlink()\n"
               " p=p.parent\n"
               " while p!=root:\n"
               "  try:p.rmdir()\n"
               "  except OSError:break\n"
               "  p=p.parent\n")
    install = ("set -e\n/root/miniconda3/bin/python -c " + shlex.quote(pruning)
               + "\ntar -xzf " + shlex.quote(location + "/source.tar.gz")
               + " -C " + shlex.quote(REMOTE_ROOT)
               + "\ncp " + shlex.quote(location + "/manifest.json")
               + " " + shlex.quote(REMOTE_ROOT + "/source_manifest.json.tmp")
               + "\nmv " + shlex.quote(REMOTE_ROOT + "/source_manifest.json.tmp")
               + " " + shlex.quote(REMOTE_ROOT + "/source_manifest.json"))
    command = ("flock -n -E 75 " + shlex.quote(CPU_LOCK) + " flock -n -E 75 "
               + shlex.quote(GPU_LOCK) + " sh -c " + shlex.quote(install))
    code = execute(client, command, timeout)
    print(json.dumps({"root": REMOTE_ROOT, "stage": manifest["stage"],
                      "deployment_id": manifest["deployment_id"], "files": len(names),
                      "archive_sha256": archive_sha, "installed": code == 0,
                      "source_snapshot": location}))
    return code


def status_command() -> str:
    script = ("import hashlib,importlib.metadata,json,os;from pathlib import Path;"
              f"root=Path({REMOTE_ROOT!r});p=root/'source_manifest.json';"
              "v=os.statvfs(root);print(json.dumps({'root':str(root),"
              "'source_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest() "
              "if p.exists() else None,"
              "'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),"
              "'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),"
              "'free_disk_bytes':v.f_bavail*v.f_frsize,"
              "'torch_version':importlib.metadata.version('torch')}))")
    return shlex.quote(REMOTE_ROOT + "/.venv/bin/python") + " -c " + shlex.quote(script)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".local/autodl-v1.json")
    parser.add_argument("--timeout", type=float, default=120)
    actions = parser.add_subparsers(dest="action", required=True)
    sync = actions.add_parser("sync")
    sync.add_argument("--paths", nargs="+")
    command = actions.add_parser("exec")
    command.add_argument("command")
    actions.add_parser("status")
    pull = actions.add_parser("pull")
    pull.add_argument("remote")
    pull.add_argument("local", type=Path)
    args = parser.parse_args()
    client = connect(args.config)
    try:
        if args.action == "sync":
            return deploy(client, args.paths, args.timeout)
        if args.action in ("exec", "status"):
            return execute(client, status_command() if args.action == "status" else args.command,
                           args.timeout)
        args.local.parent.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            sftp.get(args.remote, str(args.local))
        print(json.dumps({"saved": str(args.local), "bytes": args.local.stat().st_size}))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
