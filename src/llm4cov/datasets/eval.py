from pathlib import Path
from typing import Any

from llm4cov.datasets.types import CovExpectation, CovResult, LlmGenTbContext
from llm4cov.eda_client.remote_sync import LOCAL_TMP_DIR


def eval_cov_result_against_expectations(
    context: LlmGenTbContext,
    cov_result_raw: dict[str, Any],
) -> CovResult:
    """Evaluate raw coverage result against expectations in context.misc["targets"]."""
    cov_result = CovResult(id=context.id)
    status = cov_result_raw.get("status", "xrun_failed")
    cov_result.is_pass_xrun = status != "xrun_failed"
    cov_result.has_coverage = status == "success"
    if not cov_result.has_coverage:
        return cov_result
    exp: list[CovExpectation] = context.misc.get("targets", [])

    assert "cov_info" in cov_result_raw, "Missing 'cov_info' in result data"
    assert "summary" in cov_result_raw["cov_info"], "Missing 'summary' in cov_info"
    cov_info: list[dict[str, float | str | None]] = cov_result_raw["cov_info"]["summary"]
    assert any(
        (m["name"] == context.dut_top_module_name) and (m["level"] == 0) for m in cov_info
    ), f"Missing DUT module {context.dut_top_module_name} in cov_info"
    dut_cov_info = next(
        m for m in cov_info if (m["name"] == context.dut_top_module_name) and (m["level"] == 0)
    )
    assert "Overall Average" in dut_cov_info and isinstance(
        dut_cov_info["Overall Average"], float
    ), "Missing 'Overall Average' in DUT cov_info"
    cov_result.overall_coverage = dut_cov_info["Overall Average"]
    cov_result.misc = {k: v for k, v in dut_cov_info.items() if k not in ("name", "level")}
    # Check targets
    all_pass = True
    for target in exp:
        inst_name = target.inst_name
        if inst_name == context.dut_top_instance_name:
            # Top level is named by module name
            inst_name = context.dut_top_module_name
            tgt_cov_info = dut_cov_info
        else:
            assert any(m["name"] == inst_name for m in cov_info), (
                f"Missing instance {inst_name} in DUT cov_info "
                f"when checking target {target}, item {context.id}"
            )
            tgt_cov_info = next(m for m in cov_info if m["name"] == inst_name)
        if target.metric != "Overall Average":  # non-overall can be missing
            if target.metric not in tgt_cov_info:
                continue
            if tgt_cov_info[target.metric] is None:
                continue
        assert target.metric in tgt_cov_info, (
            f"Missing metric {target.metric} in instance {inst_name} cov_info"
        )
        actual_cov = tgt_cov_info[target.metric]
        if (not isinstance(actual_cov, float)) or actual_cov * 100 < target.target_percentage:
            all_pass = False
            break
    cov_result.is_pass_targets = all_pass
    return cov_result


def display_eval_stats(results_flat: list[tuple[str, str, CovResult]]) -> dict[str, Any]:
    # Format: list[(groupby_id, run_id, cov_result)]
    best_results: dict[str, CovResult] = {}
    best_rids: dict[str, str] = {}
    for gid, rid, res in results_flat:
        gid = str(Path(gid).relative_to(LOCAL_TMP_DIR))
        if gid not in best_results:
            best_results[gid] = res
            best_rids[gid] = rid
        else:
            if res > best_results[gid]:
                best_results[gid] = res
                best_rids[gid] = rid
    results = list(best_results.values())
    total_items = len(results)
    if total_items == 0:
        print("No results to evaluate.")
        return {"total_items": 0}  # Nothing to report
    pass_xrun = sum(1 for r in results if r.is_pass_xrun)
    have_coverage = sum(1 for r in results if r.has_coverage)
    pass_targets = sum(1 for r in results if r.is_pass_targets)
    mean_overall_cov = (
        sum(r.overall_coverage for r in results) / total_items if total_items > 0 else 0.0
    )
    mean_overall_cov_has_cov = (
        sum(r.overall_coverage for r in results if r.has_coverage) / have_coverage
        if have_coverage > 0
        else 0.0
    )
    non_agentic_items = [x for x in results if x.id.startswith("cvdp_copilot_")]
    agentic_items = [x for x in results if x.id.startswith("cvdp_agentic_")]
    pass_targets_non_agentic = sum(1 for r in non_agentic_items if r.is_pass_targets)
    pass_targets_agentic = sum(1 for r in agentic_items if r.is_pass_targets)

    label_width = 30

    print("\n=== Evaluation Summary: ===")

    print(
        f"{'Pass xrun':<{label_width}}: "
        f"{pass_xrun / total_items * 100:.1f}%  "
        f"({pass_xrun} / {total_items})"
    )

    print(
        f"{'Have coverage':<{label_width}}: "
        f"{have_coverage / total_items * 100:.1f}%  "
        f"({have_coverage} / {total_items})"
    )

    print(
        f"{'Pass coverage targets':<{label_width}}: "
        f"{pass_targets / total_items * 100:.1f}%  "
        f"({pass_targets} / {total_items})"
    )

    print(f"{'Mean overall cov':<{label_width}}: {mean_overall_cov * 100:.1f}%")
    print(f"{'Mean overall cov (has cov)':<{label_width}}: {mean_overall_cov_has_cov * 100:.1f}%")
    if len(non_agentic_items) > 0:
        print(
            f"{'Pass targets (non-agentic)':<{label_width}}: "
            f"{pass_targets_non_agentic / len(non_agentic_items) * 100:.1f}%  "
            f"({pass_targets_non_agentic} / {len(non_agentic_items)})"
        )
    if len(agentic_items) > 0:
        print(
            f"{'Pass targets (agentic)':<{label_width}}: "
            f"{pass_targets_agentic / len(agentic_items) * 100:.1f}%  "
            f"({pass_targets_agentic} / {len(agentic_items)})"
        )
    print("=========================================\n")

    best_results_output = {
        str(k): {
            "run_id": best_rids[k],
            "is_pass_xrun": v.is_pass_xrun,
            "has_coverage": v.has_coverage,
            "overall_coverage": v.overall_coverage,
            "is_pass_targets": v.is_pass_targets,
            "misc": v.misc,
        }
        for k, v in best_results.items()
    }
    ret = {
        "total_items": total_items,
        "pass_xrun": pass_xrun / total_items if total_items > 0 else 0.0,
        "have_coverage": have_coverage / total_items if total_items > 0 else 0.0,
        "pass_targets": pass_targets / total_items if total_items > 0 else 0.0,
        "mean_overall_cov": mean_overall_cov,
        "mean_overall_cov_has_cov": mean_overall_cov_has_cov,
        "pass_targets_non_agentic": (
            pass_targets_non_agentic / len(non_agentic_items) if len(non_agentic_items) > 0 else 0.0
        ),
        "pass_targets_agentic": (
            pass_targets_agentic / len(agentic_items) if len(agentic_items) > 0 else 0.0
        ),
        "best_results_output": best_results_output,
    }
    return ret
