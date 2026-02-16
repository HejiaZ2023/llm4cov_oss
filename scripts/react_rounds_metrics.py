import argparse
import json
import re
from pathlib import Path
from typing import Any

ROUND_INPUT_RE = re.compile(r"react_eval_round_inputs_(\d+)\.json$")


def _extract_post_round_best(
    round_info: list[dict[str, Any]], round_index: int
) -> dict[str, Any] | None:
    rounds: dict[int, dict[str, Any]] = {}
    final_best: dict[str, Any] | None = None
    for entry in round_info:
        if "round_index" in entry:
            rounds[int(entry["round_index"])] = entry
        elif "final_best_cov_result" in entry:
            final_best = entry["final_best_cov_result"]

    next_entry = rounds.get(round_index + 1)
    if next_entry and "best_cov_result" in next_entry:
        ret = next_entry["best_cov_result"]
        assert isinstance(ret, dict)
        return ret
    if final_best is not None:
        return final_best
    if rounds:
        ret = rounds[max(rounds)]["best_cov_result"]
        assert isinstance(ret, dict)
        return ret
    return None


def _last_round_index(round_info: list[dict[str, Any]]) -> int | None:
    last_idx = None
    for entry in round_info:
        if "round_index" in entry:
            last_idx = int(entry["round_index"])
    return last_idx


def _has_error(round_info: list[dict[str, Any]]) -> bool:
    return any("error" in entry for entry in round_info)


def _bool_metric_summary(values: list[bool], pass_k: int) -> dict[str, float]:
    if not values:
        return {"@1": 0.0, f"@{pass_k}": 0.0}
    pass_at_1 = sum(1 for v in values if v) / len(values)
    pass_at_k = 1.0 if any(values) else 0.0
    return {"@1": pass_at_1, f"@{pass_k}": pass_at_k}


def _scan_round_inputs(run_dir: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    contexts: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for path in run_dir.rglob("react_eval_round_inputs_*.json"):
        match = ROUND_INPUT_RE.search(path.name)
        if not match:
            continue
        repeat_index = int(match.group(1))
        context_id = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        contexts.setdefault(context_id, {})[repeat_index] = data
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize pass rates at different react_round settings from saved inputs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing react_eval_round_inputs_*.json files.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="Max react round to report (inclusive). Defaults to max found in inputs.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=5,
        help="k for pass@k (default: 5, clamped by available samples).",
    )
    args = parser.parse_args()

    contexts = _scan_round_inputs(args.run_dir)
    if not contexts:
        print("No react_eval_round_inputs_*.json files found.")
        return

    max_found_round = -1
    total_samples = 0
    for repeats in contexts.values():
        for round_info in repeats.values():
            total_samples += 1
            for entry in round_info:
                if "round_index" in entry:
                    max_found_round = max(max_found_round, int(entry["round_index"]))
    max_round = args.max_round if args.max_round is not None else max_found_round
    if max_round < 0:
        print("No round indices found in inputs.")
        return

    for max_round_idx in range(max_round + 1):
        pass_xrun_by_task: dict[str, list[bool]] = {}
        pass_targets_by_task: dict[str, list[bool]] = {}
        overall_cov_values: list[float] = []
        missing = 0

        for context_id, repeats in contexts.items():
            for round_info in repeats.values():
                cov = _extract_post_round_best(round_info, max_round_idx)
                if cov is None:
                    missing += 1
                    continue
                overall_cov_values.append(float(cov.get("overall_coverage", 0.0)))
                pass_xrun_by_task.setdefault(context_id, []).append(
                    bool(cov.get("is_pass_xrun", False))
                )
                pass_targets_by_task.setdefault(context_id, []).append(
                    bool(cov.get("is_pass_targets", False))
                )

        pass_k = min(args.pass_k, max((len(v) for v in pass_xrun_by_task.values()), default=1))
        pass_xrun_scores = [
            _bool_metric_summary(values, pass_k) for values in pass_xrun_by_task.values()
        ]
        pass_targets_scores = [
            _bool_metric_summary(values, pass_k) for values in pass_targets_by_task.values()
        ]

        def _avg(scores: list[dict[str, float]], pass_k: int) -> dict[str, float]:
            if not scores:
                return {"@1": 0.0, f"@{pass_k}": 0.0}
            return {
                "@1": sum(v["@1"] for v in scores) / len(scores),
                f"@{pass_k}": sum(v[f"@{pass_k}"] for v in scores) / len(scores),
            }

        pass_xrun = _avg(pass_xrun_scores, pass_k)
        pass_targets = _avg(pass_targets_scores, pass_k)
        avg_overall_cov = (
            sum(overall_cov_values) / len(overall_cov_values) if overall_cov_values else 0.0
        )

        print(
            f"max_round={max_round_idx} "
            f"pass_xrun: Pass@1= {pass_xrun['@1'] * 100:.1f}% "
            f"Pass@{pass_k}= {pass_xrun[f'@{pass_k}'] * 100:.1f}% "
            f"pass_targets: Pass@1= {pass_targets['@1'] * 100:.1f}% "
            f"Pass@{pass_k}= {pass_targets[f'@{pass_k}'] * 100:.1f}% "
            f"avg_overall_cov={avg_overall_cov * 100:.1f}% "
            f"samples={total_samples} missing={missing}"
        )

    print("\nRound improvements (overall_coverage)")
    for round_idx in range(1, max_round + 1):
        improved = 0
        total = 0
        for repeats in contexts.values():
            for round_info in repeats.values():
                prev_cov = _extract_post_round_best(round_info, round_idx - 1)
                curr_cov = _extract_post_round_best(round_info, round_idx)
                if prev_cov is None or curr_cov is None:
                    continue
                prev_val = float(prev_cov.get("overall_coverage", 0.0))
                curr_val = float(curr_cov.get("overall_coverage", 0.0))
                total += 1
                if curr_val > prev_val:
                    improved += 1
        pct = (improved / total * 100.0) if total else 0.0
        print(f"round={round_idx}: improved={improved}/{total} ({pct:.1f}%)")

    print("\nErrors by round")
    errors_by_round: dict[int, int] = {i: 0 for i in range(max_round + 1)}
    total_errors = 0
    for repeats in contexts.values():
        for round_info in repeats.values():
            if not _has_error(round_info):
                continue
            error_round = _last_round_index(round_info)
            if error_round is None or error_round > max_round:
                continue
            errors_by_round[error_round] += 1
            total_errors += 1
    for round_idx in range(max_round + 1):
        print(f"round={round_idx}: errors={errors_by_round[round_idx]}")
    print(f"errors_total={total_errors}")


if __name__ == "__main__":
    main()
