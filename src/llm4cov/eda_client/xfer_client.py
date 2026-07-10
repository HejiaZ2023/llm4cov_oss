"""统一传输层 EDA 覆盖率客户端。

- local: 直接把 job 写进本地 xfer 目录(paladin eval,同机,零网络)。
- sftp : 经 relay 容器 sftp 投/取(brev 训练,跨主机走 tailscale;纯文件、零 exec)。

两条路汇聚到同一个 xfer_watcher;传输由 EDA_TRANSPORT 或 server 推断。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

_LOCAL_SERVERS = {"", "local", "paladin", "paladin_centos"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _encode(content: str | bytes) -> bytes:
    return content.encode() if isinstance(content, str) else content


def _attach_relay_timing(result: dict[str, Any], raw: bytes | str | None) -> dict[str, Any]:
    """Attach watcher-side timing when the relay published it alongside result.json."""
    if raw is None:
        return result
    try:
        timing = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return result
    if isinstance(timing, dict):
        result["relay_timing"] = timing
    return result


class Transport:
    """传输抽象:submit job / poll result / cleanup。"""

    def submit(self, job_id: str, inputs: dict[str, bytes], manifest: dict[str, Any]) -> None:
        raise NotImplementedError

    def poll_result(self, job_id: str, timeout: float) -> dict[str, Any]:
        raise NotImplementedError

    def cleanup(self, job_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalTransport(Transport):
    """本地文件系统传输(eval 同机用)。"""

    def __init__(self, xfer_dir: str) -> None:
        self.x = Path(xfer_dir)

    def submit(self, job_id: str, inputs: dict[str, bytes], manifest: dict[str, Any]) -> None:
        stg = self.x / "incoming" / f"{job_id}.staging"
        (stg / "inputs").mkdir(parents=True, exist_ok=True)
        for name, content in inputs.items():
            (stg / "inputs" / name).write_bytes(content)
        (stg / "manifest.json").write_text(json.dumps(manifest))
        os.rename(stg, self.x / "incoming" / job_id)  # 原子发布

    def poll_result(self, job_id: str, timeout: float) -> dict[str, Any]:
        done = self.x / "results" / job_id / ".done"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if done.exists():
                text = (self.x / "results" / job_id / "result.json").read_text()
                result: dict[str, Any] = json.loads(text)
                timing_path = self.x / "results" / job_id / "relay_timing.json"
                return _attach_relay_timing(
                    result,
                    timing_path.read_text() if timing_path.is_file() else None,
                )
            time.sleep(1)
        raise TimeoutError(f"EDA result timeout: {job_id}")

    def cleanup(self, job_id: str) -> None:
        for p in (self.x / "incoming" / job_id, self.x / "results" / job_id):
            shutil.rmtree(p, ignore_errors=True)


class SftpTransport(Transport):
    """SFTP 传输(brev 跨主机经 relay 用);仅文件操作,无远程 exec。"""

    def __init__(self, host: str, port: int, user: str, key: str) -> None:
        import paramiko  # type: ignore[import-untyped]

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            host,
            port=port,
            username=user,
            key_filename=key,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
        )
        self.sftp = self.client.open_sftp()

    def _mkdirs(self, path: str) -> None:
        cur = ""
        for part in path.strip("/").split("/"):
            cur = f"{cur}/{part}" if cur else part
            with contextlib.suppress(IOError):
                self.sftp.mkdir(cur)

    def submit(self, job_id: str, inputs: dict[str, bytes], manifest: dict[str, Any]) -> None:
        base = f"incoming/{job_id}.staging"
        self._mkdirs(f"{base}/inputs")
        for name, content in inputs.items():
            with self.sftp.open(f"{base}/inputs/{name}", "wb") as f:
                f.write(content)
        with self.sftp.open(f"{base}/manifest.json", "wb") as f:
            f.write(json.dumps(manifest).encode())
        self.sftp.posix_rename(base, f"incoming/{job_id}")  # 原子发布

    def poll_result(self, job_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.sftp.stat(f"results/{job_id}/.done")
            except OSError:
                time.sleep(2)
                continue
            with self.sftp.open(f"results/{job_id}/result.json", "rb") as f:
                result: dict[str, Any] = json.loads(f.read())
            try:
                with self.sftp.open(f"results/{job_id}/relay_timing.json", "rb") as f:
                    relay_timing = f.read()
            except OSError:
                relay_timing = None
            return _attach_relay_timing(result, relay_timing)
        raise TimeoutError(f"EDA result timeout: {job_id}")

    def _rmtree(self, path: str) -> None:
        try:
            entries = self.sftp.listdir(path)
        except OSError:
            return
        for entry in entries:
            child = f"{path}/{entry}"
            try:
                self.sftp.remove(child)
            except OSError:
                self._rmtree(child)
        with contextlib.suppress(IOError):
            self.sftp.rmdir(path)

    def cleanup(self, job_id: str) -> None:
        for d in (f"incoming/{job_id}", f"results/{job_id}"):
            self._rmtree(d)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sftp.close()
            self.client.close()


def make_transport(server: str) -> Transport:
    transport = os.environ.get("EDA_TRANSPORT", "").lower()
    if not transport:
        transport = "local" if server in _LOCAL_SERVERS else "sftp"
    if transport == "local":
        return LocalTransport(os.environ.get("EDA_XFER_DIR", "/mnt/raid0_ssd/eda/xfer"))
    return SftpTransport(
        os.environ.get("EDA_SFTP_HOST", server),
        int(os.environ.get("EDA_SFTP_PORT", "2222")),
        os.environ.get("EDA_SFTP_USER", "gpujobs"),
        os.environ.get("EDA_SFTP_KEY", os.path.expanduser("~/.ssh/brev_eda_sftp")),
    )


def submit_cov_job(
    server: str,
    eda_repo_dir: str,
    context: Any,
    tb_file: Any,
    skip_detail: bool = False,
    timeout: int = 600,
) -> dict[str, Any]:
    """构 job → 选传输 → 提交 → 轮询结果 → 清理;返回 result.json 解析后的 dict。"""
    started_monotonic = time.monotonic()
    submitted_at_unix = time.time()
    dut = context.dut_top_module_name
    sv_files = [f.name for f in context.rtl_files]
    inputs = {f.name: _encode(f.content) for f in context.rtl_files}
    inputs[tb_file.name] = _encode(tb_file.content)
    job_id = f"{dut}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "dut": dut,
        "tb": tb_file.name,
        "sv_files": sv_files,
        "cov_type": "all",
        "metrics": ["overall", "code", "fsm", "functional", "toggle", "block", "assertion"],
        "timeout": timeout,
        "skip_detail": skip_detail,
        "inputs_sha256": {n: _sha(c) for n, c in inputs.items()},
        "eda_repo_dir": eda_repo_dir,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(submitted_at_unix)),
        "submitted_at_unix": submitted_at_unix,
    }
    connect_started = time.monotonic()
    transport = make_transport(server)
    connect_seconds = time.monotonic() - connect_started
    submit_seconds = 0.0
    wait_result_seconds = 0.0
    cleanup_seconds = 0.0
    try:
        submit_started = time.monotonic()
        transport.submit(job_id, inputs, manifest)
        submit_seconds = time.monotonic() - submit_started
        wait_started = time.monotonic()
        result = transport.poll_result(job_id, timeout * 3)
        wait_result_seconds = time.monotonic() - wait_started
        cleanup_started = time.monotonic()
        transport.cleanup(job_id)
        cleanup_seconds = time.monotonic() - cleanup_started
    finally:
        transport.close()
    fetched_at_unix = time.time()
    result.update(
        {
            "job_id": job_id,
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fetched_at_unix)),
            "fetched_at_unix": fetched_at_unix,
            "xfer_timing": {
                "transport": type(transport).__name__,
                "submitted_at_unix": submitted_at_unix,
                "transport_connect_seconds": connect_seconds,
                "submit_seconds": submit_seconds,
                "wait_result_seconds": wait_result_seconds,
                "cleanup_seconds": cleanup_seconds,
                "total_seconds": time.monotonic() - started_monotonic,
            },
        }
    )
    return result
