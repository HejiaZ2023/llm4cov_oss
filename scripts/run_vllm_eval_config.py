import argparse
import contextlib
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from io import BufferedWriter
from pathlib import Path
from typing import Any, NoReturn

_active_vllm_proc: subprocess.Popen | None = None


def _sigint_handler(signum: Any, frame: Any) -> NoReturn:
    global _active_vllm_proc
    print("\nSIGINT received, cleaning up...")

    if _active_vllm_proc and _active_vllm_proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(_active_vllm_proc.pid, signal.SIGINT)
            _active_vllm_proc.wait(timeout=10)

    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _sigint_handler)


def _split_env_prefix(cmd: str) -> tuple[dict[str, str], str]:
    tokens = shlex.split(cmd)
    env: dict[str, str] = {}
    rest = ""
    for i, token in enumerate(tokens):
        if "=" in token and not token.startswith("-") and token.index("=") > 0:
            key, value = token.split("=", 1)
            env[key] = value
        else:
            rest = " ".join(tokens[i:])
            break
    return env, rest


def get_vllm_port_from_command(cmd: str) -> int:
    tokens = shlex.split(cmd)
    assert _is_vllm_serve_command(cmd)

    port = 8000  # default port
    args = tokens[2:]
    it = iter(args)
    for token in it:
        if token == "--port":
            port = int(next(it))

    return port


def _is_vllm_serve_command(cmd: str) -> bool:
    tokens = shlex.split(cmd)
    if not tokens:
        return False
    return tokens[0] == "vllm" and len(tokens) > 1 and tokens[1] == "serve"


def _is_eval_command(cmd: str) -> bool:
    tokens = shlex.split(cmd)
    if not tokens:
        return False
    if tokens[0] == "python":
        start = 1
    elif tokens[0] == "uv" and len(tokens) > 2 and tokens[1] == "run":
        start = 2
    else:
        return False

    return not (len(tokens) <= start or tokens[start] != "scripts/batch_query_eval.py")


def _parse_bash_to_commands(bash_path: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    current_item: dict[str, Any] = dict()
    with bash_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            env_dict, cmd = _split_env_prefix(line)
            if _is_vllm_serve_command(cmd):
                if current_item:
                    configs.append(current_item)
                    current_item = dict()
                port = get_vllm_port_from_command(cmd)
                current_item["vllm_cmd"] = cmd
                current_item["vllm_env"] = env_dict
                current_item["port"] = port
                current_item["eval_items"] = []
            elif _is_eval_command(cmd):
                if not current_item:
                    raise ValueError(f"No vllm serve line before eval at line {line_no}")
                current_item["eval_items"].append({"eval_cmd": cmd, "eval_env": env_dict})
            else:
                raise ValueError(f"Unrecognized command at line {line_no}: {line}")
        if current_item and len(current_item.get("eval_items", [])) > 0:
            configs.append(current_item)
    return configs


def _launch_vllm(cmd: str, env: dict[str, str], log_f: BufferedWriter) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        start_new_session=True,
        env={**os.environ, **env},
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    return proc


def _wait_for_ready(port: int, timeout_s: float, proc: subprocess.Popen[bytes]) -> None:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    time.sleep(1)
                    return
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_err = exc
        time.sleep(1)
    raise RuntimeError(f"vLLM server did not become ready on port {port}: {last_err}")


def _shutdown_vllm(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return

    def _kill(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)

    # 1. SIGINT (graceful, same as Ctrl-C)
    _kill(signal.SIGINT)
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass

    # 2. SIGTERM (forceful but clean)
    _kill(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass

    # 3. SIGKILL (last resort)
    _kill(signal.SIGKILL)
    proc.wait()


def _run_eval(cmd: str, env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        env={**os.environ, **env},
        start_new_session=True,  # IMPORTANT
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait()
        raise


def main() -> int:
    global _active_vllm_proc
    parser = argparse.ArgumentParser(
        description="Run vLLM + batch_query_eval pairs from a bash file."
    )
    parser.add_argument(
        "--config-bash",
        type=Path,
        required=True,
        help="Bash file containing vllm serve and batch_query_eval commands.",
    )
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=600.0,
        help="Seconds to wait for vLLM readiness.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next config when a run fails.",
    )
    parser.add_argument(
        "--vllm-log-dir",
        type=Path,
        default=Path("./log/vllm").absolute(),
        help="Path to save vLLM server logs.",
    )

    args = parser.parse_args()
    records = _parse_bash_to_commands(args.config_bash)
    assert isinstance(args.vllm_log_dir, Path)
    args.vllm_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Parsed {len(records)} vLLM + eval configurations:")
    print(records)

    for index, payload in enumerate(records, start=1):
        vllm_cmd = payload["vllm_cmd"]
        vllm_env = payload["vllm_env"]
        port = payload["port"]
        eval_items = payload["eval_items"]
        vllm_log_file = args.vllm_log_dir / f"vllm_server_{index}.log"

        with vllm_log_file.open("wb") as vllm_log_f:
            print(f"==> [{index}] launching vLLM")
            proc = _launch_vllm(vllm_cmd, vllm_env, vllm_log_f)
            _active_vllm_proc = proc
            try:
                _wait_for_ready(port, args.ready_timeout_s, proc)
                print(f"==> [{index}] vLLM ready on port {port}, running eval")
                for index_eval, eval_item in enumerate(eval_items, start=1):
                    eval_cmd = eval_item["eval_cmd"]
                    eval_env = eval_item["eval_env"]
                    returncode = _run_eval(eval_cmd, eval_env)
                    if returncode != 0:
                        msg = (
                            f"Eval failed for entry {index}, eval {index_eval} "
                            f"with exit code {returncode}"
                        )
                        if args.continue_on_error:
                            print(msg)
                        else:
                            raise RuntimeError(msg)
                    time.sleep(1)
            finally:
                _active_vllm_proc = None
                print(f"==> [{index}] shutting down vLLM")
                _shutdown_vllm(proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
