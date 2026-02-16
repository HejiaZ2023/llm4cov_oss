from llm4cov.datasets.types import LlmGenTbContext

COV_SYSTEM_PROMPT = """
You are a testbench expert.

Your job is to write a SystemVerilog testbench that maximizes coverage of the given RTL project.

OUTPUT REQUIREMENTS:

1. You MUST explicitly state the filename for the testbench in plain text using format:
     filename: tb_xxxx.sv
2. After stating the filename, you MUST output the complete testbench
    inside a fenced SystemVerilog code block:
```systemverilog
module tb_example;
  ...
endmodule
```
3. The SystemVerilog code MUST NOT include any checks or assertions on outputs.
Only generate stimulus to exercise as many RTL paths as possible.
4. Use the provided RTL context and instructions to produce the highest possible coverage.
5. Put NOTHING except the code inside the fenced code block.
6. Do NOT wrap the filename inside a code block.

OVERALL OUTPUT ORDER:
1. Plain-text filename line
2. SystemVerilog fenced code block
"""

COV_USER_TEMPLATE = """
<context>
{context}
</context>
<instructions>
{instructions}
</instructions>
Write a SystemVerilog testbench that maximizes coverage of the given RTL project.
"""

PATCH_SYSTEM_PROMPT = """
You are a patch hunk generator.

Your task is to generate edit hunks for EXACTLY ONE existing file.
The file path is known externally and MUST NOT appear in your output.

=====================
OUTPUT CONTRACT
=====================

- Output ONLY patch hunks.
- Do NOT output:
  - file headers
  - begin/end markers
  - explanations, comments, or markdown
- If no edits are required, output exactly:
  NO_CHANGES

=====================
HUNK FORMAT
=====================

Each hunk represents ONE contiguous edit and must follow this structure:

1. An opening line:
   @@

2. Pre-context lines
   - 1-3 lines that appear immediately BEFORE the change
   - Each line MUST start with a single space: " "

3. Old code lines (to be removed)
   - Each line MUST start with "-"

4. New code lines (to be added)
   - Each line MUST start with "+"

5. Post-context lines
   - 1-3 lines that appear immediately AFTER the change
   - Each line MUST start with a single space: " "

=====================
CONTEXT RULES
=====================

- Pre-context and post-context MUST match the file contents EXACTLY.
- Include enough context to uniquely identify where the change applies.
- Prefer the smallest amount of context that is unambiguous.
- Do NOT duplicate context unnecessarily:
  - If two changes are very close, merge them into one hunk.
- Do NOT invent or paraphrase surrounding code.

=====================
VALIDITY CONSTRAINTS
=====================

- Every non-empty output line MUST start with exactly one of:
    " "  (context)
    "-"  (old code)
    "+"  (new code)
    "@"  (hunk header)
- The prefix character MUST be the first character on the line.
- Do NOT include tabs or extra indentation before the prefix.
- Do NOT include absolute or relative file paths.

=====================
GOAL
=====================

Given:
1) the full current contents of the target file, and
2) a natural-language description of desired changes,

produce patch hunks that, when applied in order, implement the requested edits
correctly and minimally.
"""


def build_initial_prompt_from_context(
    context: LlmGenTbContext,
) -> list[dict[str, str]]:
    """Build prompt string from LlmGenTbContext and a prompt template."""
    rtl_contents = "\n\n".join(f"// File: {f.name}\n{f.content}" for f in context.rtl_files)
    spec_contents = (
        "\n\n".join(f"// File: {f.name}\n{f.content}" for f in context.spec_files)
        if context.spec_files
        else "// No specification files provided."
    )

    user_prompt: str = COV_USER_TEMPLATE.format(
        context=rtl_contents + spec_contents, instructions=context.instructions
    )
    return [
        {"role": "system", "content": COV_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_react_followup_prompt(
    *,
    parse_status: dict[str, str],
    apply_status: dict[str, str] | None,
    eda_status: dict[str, str] | None,
    tb_content: str | None,
    instruction: str,
    is_single_message: bool = False,
) -> list[dict[str, str]]:
    """Build follow-up messages containing tool feedback and routed instruction."""
    messages: list[dict[str, str]] = []

    parse_lines = ["Response parse result:"] + [f"- {k}: {v}" for k, v in parse_status.items()]
    messages.append({"role": "user", "content": "\n".join(parse_lines)})

    if apply_status:
        apply_lines = ["Patch apply result:"] + [f"- {k}: {v}" for k, v in apply_status.items()]
        messages.append({"role": "user", "content": "\n".join(apply_lines)})

    if tb_content is not None and parse_status.get("type") == "patch":
        messages.append(
            {"role": "user", "content": f"Current testbench contents:\n{tb_content.rstrip()}"}
        )

    if eda_status:
        eda_lines = ["EDA result:"] + [f"- {k}: {v}" for k, v in eda_status.items()]
        messages.append({"role": "user", "content": "\n".join(eda_lines)})

    messages.append(
        {"role": "user", "content": f"Instruction for next round:\n{instruction.strip()}"}
    )
    if is_single_message:
        combined_content = "\n\n".join(msg["content"] for msg in messages)
        return [{"role": "user", "content": combined_content}]
    return messages
