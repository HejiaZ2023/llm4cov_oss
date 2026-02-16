import hashlib
from functools import total_ordering
from typing import Any

from pydantic import BaseModel


class DataFile(BaseModel):
    name: str
    content: str


class DataContext(BaseModel):
    id: str
    rtl_files: list[DataFile]
    spec_files: list[DataFile]
    dataset_name: str
    rtl_tokens: int
    potential_top: list[str] = []
    misc: dict[str, Any] = {}


class LlmGenTbContext(DataContext):
    dut_top_module_name: str
    dut_top_instance_name: str
    instructions: str


@total_ordering
class CovResult(BaseModel):
    id: str
    is_pass_xrun: bool = False
    has_coverage: bool = False
    overall_coverage: float = 0.0
    is_pass_targets: bool = False
    misc: dict[str, Any] = {}

    def _cmp_key(self) -> tuple[bool, bool, bool, float]:
        return (
            self.is_pass_xrun,
            self.has_coverage,
            self.is_pass_targets,
            self.overall_coverage,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CovResult):
            return NotImplemented
        return self._cmp_key() == other._cmp_key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CovResult):
            return NotImplemented
        return self._cmp_key() < other._cmp_key()


class CovExpectation(BaseModel):
    inst_name: str
    metric: str
    target_percentage: float


def _hash_to_float(key: str) -> float:
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def data_context_to_llm_gen_tb_context(context: DataContext) -> LlmGenTbContext:
    """Convert a DataContext to LlmGenTbContext"""
    if isinstance(context, LlmGenTbContext):
        # Already the right type, like in CVDP-EcoV dataset
        return context
    dut_top_module_name = context.potential_top[0]  # Assume single top module after filtering

    # In synthetic data, we should have some variation in instance naming style.
    r = _hash_to_float(context.id)
    dut_top_instance_name = f"inst_{dut_top_module_name}"
    if r < 0.33:
        dut_top_instance_name = "dut"
    elif r < 0.66:
        dut_top_instance_name = "uut"

    instructions = (
        "Generate a SystemVerilog testbench for the given RTL design. "
        f"The top module is '{dut_top_module_name}', "
        f"and its instance name in the testbench should be '{dut_top_instance_name}'. "
        "Target coverage only in your testbench: "
        "Try to achieve as high coverage as possible, including line, toggle, FSM, branch, etc. "
        "Also, don't include assertion statements in the testbench; they'll be handled separately."
    )

    return LlmGenTbContext(
        id=context.id,
        rtl_files=context.rtl_files,
        spec_files=context.spec_files,
        dataset_name=context.dataset_name,
        rtl_tokens=context.rtl_tokens,
        potential_top=context.potential_top,
        misc=context.misc,
        instructions=instructions,
        dut_top_module_name=dut_top_module_name,
        dut_top_instance_name=dut_top_instance_name,
    )
