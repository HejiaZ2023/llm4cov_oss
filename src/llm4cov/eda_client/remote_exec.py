#!/usr/bin/env python3
"""
Client-side helper for running llm4cov_eda.worker.get_cov_result remotely.
Now includes local temp file collision handling and cleanup.
llm4cov_eda is included as submodule at submodules/llm4cov_eda/src/llm4cov_eda.
"""

import contextlib
import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from llm4cov.datasets.types import DataFile, LlmGenTbContext
from llm4cov.eda_client.remote_sync import clear_remote_dir, prepare_local_dir, sync_dir_to_remote


def run_remote_cov_job_pipeline(
    server: str,
    eda_repo_dir: str,
    context: LlmGenTbContext,
    tb_file: DataFile,
    skip_detail: bool = False,
    timeout: int = 600,
) -> dict[str, Any]:
    """Prepare local dir, sync to remote, run remote cov job, fetch result, and cleanup."""
    # --- 1️⃣ Prepare local dir ---
    local_work_dir = prepare_local_dir(context, tb_file)

    # --- 2️⃣ Sync to remote ---
    remote_work_dir = sync_dir_to_remote(local_work_dir, server)

    # --- 3️⃣ Run remote coverage job ---
    result = run_remote_cov_job_wrapper(
        server=server,
        eda_repo_dir=eda_repo_dir,
        remote_work_dir=remote_work_dir,
        local_work_dir=local_work_dir,
        context=context,
        tb_file=tb_file,
        timeout=timeout,
        skip_detail=skip_detail,
    )

    assert "local_work_dir" not in result, "local_work_dir should not be in the result"
    result["local_work_dir"] = local_work_dir

    # --- 4️⃣ Cleanup remote dir ---
    time.sleep(0.1)  # Ensure all remote ops are done
    clear_remote_dir(remote_work_dir, server)

    return result


def run_remote_cov_job_wrapper(
    server: str,
    eda_repo_dir: str,
    remote_work_dir: str,
    local_work_dir: Path,
    context: LlmGenTbContext,
    tb_file: DataFile,
    timeout: int = 600,
    skip_detail: bool = False,
) -> dict[str, Any]:
    """Wrapper to prepare and run remote coverage job, returning coverage result dict."""
    # Prepare SV file paths
    sv_files = [f.name for f in context.rtl_files]
    tb_filename = tb_file.name
    dut_name = context.dut_top_module_name
    metrics = ["overall", "code", "fsm", "functional", "toggle", "block", "assertion"]

    # Run remote coverage job
    result = run_remote_cov_job(
        server=server,
        repo_dir=eda_repo_dir,
        work_dir=remote_work_dir,
        local_tmp_dir=local_work_dir,
        sv_files=sv_files,
        tb_file=tb_filename,
        dut_name=dut_name,
        metrics=metrics,
        timeout=timeout,
        skip_detail=skip_detail,
        cleanup=False,
    )
    return result


def run_remote_cov_job(
    server: str,
    repo_dir: str,
    work_dir: str,
    sv_files: list[str],
    tb_file: str,
    dut_name: str,
    metrics: list[str] | None = None,
    cov_type: str = "all",
    timeout: int = 600,
    outfile: str = "result.json",
    skip_detail: bool = False,
    local_tmp_dir: Path | None = None,
    cleanup: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run get_cov_result.py on remote server and fetch result.json safely."""

    # --- 0️⃣ Prepare unique local tmp file ---
    tmp_dir = Path(local_tmp_dir or Path("/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    unique_id = f"{dut_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    local_tmp = tmp_dir / f"{unique_id}_{outfile}"

    # --- 1️⃣ Build remote command ---
    sv_args = " ".join(shlex.quote(f) for f in sv_files)
    metrics_args = " ".join(shlex.quote(m) for m in (metrics or []))
    remote_cmd = (
        f"cd {shlex.quote(repo_dir)} && "
        f"uv run -m llm4cov_eda.get_cov_result "
        f"--workdir {shlex.quote(work_dir)} "
        f"--sv_files {sv_args} "
        f"--tb {shlex.quote(tb_file)} "
        f"--dut {shlex.quote(dut_name)} "
        f"--cov_type {shlex.quote(cov_type)} "
        f"--timeout {timeout} "
        f"--outfile {shlex.quote(outfile)} "
        f"{'--skip_detail ' if skip_detail else ''}"
    )
    if metrics_args:
        remote_cmd += f"--metrics {metrics_args} "

    # --- 2️⃣ Execute on remote server ---
    ssh_cmd = ["ssh", server, remote_cmd]
    if verbose:
        print(f">>> SSH: {' '.join(ssh_cmd)}")
    proc = subprocess.run(
        ssh_cmd,
        text=True,
        capture_output=True,
        timeout=timeout * 3,  # Each remote instruction takes a timeout period
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Remote worker failed with code {proc.returncode}\n"
            f"--- STDOUT ---\n{proc.stdout[-400:]}\n"
            f"--- STDERR ---\n{proc.stderr[-400:]}"
        ) from None

    # --- 3️⃣ Copy result.json back ---
    scp_cmd = ["scp", f"{server}:{work_dir}/{outfile}", str(local_tmp)]
    if verbose:
        print(f">>> SCP: {' '.join(scp_cmd)}")
    scp_proc = subprocess.run(scp_cmd, text=True, capture_output=True)
    if scp_proc.returncode != 0:
        raise RuntimeError(f"Failed to scp result.json back: {scp_proc.stderr[-400:]}") from None

    # --- 4️⃣ Parse JSON safely ---
    try:
        with local_tmp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
    except Exception as e:
        raise RuntimeError(f"Invalid or missing JSON in result.json: {e}") from None
    finally:
        if cleanup:
            with contextlib.suppress(Exception):
                local_tmp.unlink(missing_ok=True)

    # --- 5️⃣ Return parsed result with metadata ---
    data.update(
        {
            "remote_repo_dir": repo_dir,
            "remote_work_dir": work_dir,
            "local_tmp_used": str(local_tmp),
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return data
