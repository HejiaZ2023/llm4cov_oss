import subprocess
import time
import uuid
from pathlib import Path

from llm4cov.eda_client.remote_exec import run_remote_cov_job

# --- 1️⃣ Prepare a unique remote work dir ---
uid = uuid.uuid4().hex[:6]
remote_work_dir = f"/tmp/proj_counter_{uid}"

# --- 2️⃣ Local resources to sync (assuming ./resources/ exists near this script) ---
this_file_dir = Path(__file__).resolve().parent
local_resources = this_file_dir / "resources"

# --- 3️⃣ Rsync resources to remote workdir ---
server = "wolverine.ucsd.edu"
rsync_cmd = [
    "rsync",
    "-avz",  # archive + verbose + compression
    "--delete",  # optional: mirror local directory
    f"{local_resources}/",  # trailing slash copies contents only
    f"{server}:{remote_work_dir}/",  # remote target
]
print(">>> RSYNC:", " ".join(rsync_cmd))
subprocess.run(rsync_cmd, check=True)

# --- 4️⃣ Run remote simulation worker ---
start = time.time()
res = run_remote_cov_job(
    server=server,
    repo_dir="/workspace/llm4cov",
    work_dir=remote_work_dir,
    sv_files=["counter.sv", "tb_counter.sv"],
    tb_file="tb_counter.sv",
    dut_name="counter",
    metrics=["code", "toggle", "block"],
    timeout=900,
)
runtime_client = round(time.time() - start, 3)

# --- 5️⃣ Display results ---
print("Client run time:", runtime_client)
print("✅ Coverage summary:", res["cov_info"]["summary"])
print("✅ Coverage detail:")
print(res["cov_info"]["detail"])
res.pop("cov_info")
print("response:", res)
