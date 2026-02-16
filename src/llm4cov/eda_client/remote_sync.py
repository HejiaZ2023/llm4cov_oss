import subprocess
import time
import uuid
from pathlib import Path

from llm4cov.datasets.types import DataFile, LlmGenTbContext

LOCAL_TMP_PARENT_DIR = Path(__file__).parents[3] / "eda_intermediate"
# run_date_time__uuid
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}__{str(uuid.uuid4())[:4]}"
LOCAL_TMP_DIR = LOCAL_TMP_PARENT_DIR / RUN_ID


def _allocate_local_dir(parent: Path) -> Path:
    """Allocate a unique local directory under the given parent path."""
    while True:
        candidate = parent / uuid.uuid4().hex[:6]
        try:
            candidate.mkdir(parents=True, exist_ok=False)  # On POSIX filesystems, mkdir is atomic
            return candidate
        except FileExistsError:
            # Extremely unlikely, but safe
            continue


def prepare_local_dir(context: LlmGenTbContext, tb_file: DataFile) -> Path:
    """Prepare local directory for remote sync operations."""
    local_dir = _allocate_local_dir(LOCAL_TMP_DIR / context.dataset_name / context.id)
    local_dir.mkdir(parents=True, exist_ok=True)
    for rtl_file in context.rtl_files:
        rtl_path = local_dir / rtl_file.name
        with open(rtl_path, "w") as f:
            f.write(rtl_file.content)
    tb_path = local_dir / tb_file.name
    with open(tb_path, "w") as f:
        f.write(tb_file.content)
    return local_dir


def sync_dir_to_remote(local_dir: Path, server: str, remote_dir: str | None = None) -> str:
    """Sync local directory to remote directory using rsync. Return remote dir"""
    # check given dir under local tmp dir
    assert local_dir.is_relative_to(LOCAL_TMP_PARENT_DIR), (
        f"Local dir {local_dir} is not under {LOCAL_TMP_PARENT_DIR}"
    )
    if remote_dir is None:
        relative_dir = local_dir.relative_to(LOCAL_TMP_PARENT_DIR)
        remote_dir = "/tmp/llm4cov_eda/" + str(relative_dir)

    subprocess.run(
        ["ssh", server, f"mkdir -p {remote_dir}"],
        check=True,
    )

    rsync_cmd = [
        "rsync",
        "-az",
        "--delete",
        f"{local_dir}/",
        f"{server}:{remote_dir}/",
    ]
    subprocess.run(rsync_cmd, check=True)
    return remote_dir


def clear_remote_dir(remote_dir: str, server: str) -> None:
    """Clear the remote directory."""
    ssh_cmd = [
        "ssh",
        server,
        f"rm -rf {remote_dir}",
    ]
    subprocess.run(ssh_cmd, check=True)
