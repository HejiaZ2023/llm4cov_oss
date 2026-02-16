VANILLA_PROMPT = """
You are a verification expert.

Your task is NOT to modify the testbench.
Your task is to INFER AND SYNTHESIZE the reasoning that would naturally lead
to the given testbench, given the RTL design and verification objective.

Context:
- You are given:
  (1) the RTL design
  (2) the generated testbench
- This is a vanilla generation run:
  there is no coverage feedback, no simulator failure, and no iterative correction.
- The testbench already exists and is considered correct.

Infer the internal reasoning that would lead a verification engineer
to write this testbench as an initial, straightforward validation of the design,
based on the design intent and basic simulation requirements.

The reasoning should be written as if it preceded the testbench generation,
even though the testbench is already known to be correct.

Reasoning requirements:
1. The reasoning MUST be consistent with the RTL behavior and design requirements.
2. The reasoning MUST describe the intent behind instantiating the DUT,
   selecting a bounded simulation duration, and terminating the simulation.
3. The reasoning MUST be framed as an initial, first-pass validation
   rather than a response to any prior outcome.
4. The reasoning MUST remain aligned with the structure and behavior
   of the given testbench.

Style requirements:
- Use concise, technical, verification-oriented language.
- Neutral, non-agentic tone (no exploration or decision branching).
- No self-correction, no speculation, no backtracking.
- No mention of being an AI or model.
- No mention of future improvements.

Output format:
- Output ONLY the reasoning.
- Do NOT repeat the testbench code.
- Do NOT output filenames or code blocks.
- Use short paragraphs or numbered steps.

This reasoning will be used as synthetic supervision data
for vanilla (non-ReAct) testbench generation.
"""

REACT_XRUN_PROMPT = """
You are a verification and simulation-debugging expert.

Your task is NOT to modify the testbench.
Your task is to INFER AND SYNTHESIZE the reasoning that would naturally lead
to the given testbench update that fixes an xrun failure.

Context:
- You are given:
  (1) the RTL design
  (2) the current testbench
  (3) xrun failure feedback, including simulator logs
- The updated testbench already exists and is considered the correct action.

Infer the internal decision-making steps that would lead an expert
verification engineer to produce the given testbench update,
given the simulator failure mode and the verification objective.

The reasoning should be written as if it preceded the testbench update,
even though the update is already known to be correct.

Reasoning requirements:
1. The reasoning MUST be consistent with the xrun failure report and logs.
2. The reasoning MUST identify the root cause of the xrun failure
   (e.g., non-terminating simulation, missing $finish, infinite activity).
3. The reasoning MUST describe causal links between the failure mode
   and the specific testbench changes that resolve it.
4. The reasoning MUST NOT propose fixes that are NOT present in the testbench.
5. The reasoning MUST NOT introduce uncertainty, alternatives, or dead ends.
6. The reasoning MUST assume the testbench update is correct and intentional.

Style requirements:
- Use concise, technical, simulation- and tool-oriented language.
- No self-correction, no speculation, no backtracking.
- No mention of being an AI or model.
- No mention of future improvements.

Output format:
- Output ONLY the reasoning.
- Do NOT repeat the testbench code.
- Do NOT output filenames or code blocks.
- Use numbered steps or short paragraphs.

This reasoning will be used as synthetic ReAct-style supervision data
for xrun failure recovery.
"""

REACT_COVERAGE_PROMPT = """
You are a coverage-driven verification expert.

Your task is NOT to modify the testbench.
Your task is to INFER AND SYNTHESIZE the reasoning
    that would naturally lead to the given testbench update.

Context:
- You are given:
  (1) the RTL design
  (2) the current testbench
  (3) coverage feedback from the previous simulation
- The testbench already exists and is considered the correct action.

Infer the internal decision-making steps that would lead an expert
verification agent to produce the given testbench update,
given the coverage feedback and the verification objective.

The reasoning should be written as if it preceded the testbench update,
even though the update is already known to be correct.

Reasoning requirements:
1. The reasoning MUST be consistent with the provided coverage report.
2. The reasoning MUST explicitly reference uncovered or partially covered items
   (e.g., counter bits, toggles, blocks, FSMs).
3. The reasoning MUST describe causal links between coverage gaps
	and the specific testbench stimulus or structure that addresses them.
4. The reasoning MUST NOT propose changes that are NOT present in the testbench.
5. The reasoning MUST NOT introduce uncertainty, alternatives, or dead ends.
6. The reasoning MUST assume the testbench update is correct and intentional.

Style requirements:
- Use concise, technical, verification-oriented language.
- No self-correction, no speculation, no backtracking.
- No mention of being an AI or model.
- No mention of future improvements.

Output format:
- Output ONLY the reasoning.
- Do NOT repeat the testbench code.
- Do NOT output filenames or code blocks.
- Use numbered steps or short paragraphs.

This reasoning will be used as synthetic ReAct-style supervision data.
"""

# user prompt template: feed in synthetic messages (input )
USER_TEMPLATE = """
<synthetic_input_messages>
{synthetic_input_messages}
</synthetic_input_messages>
<synthetic_output_message>
{synthetic_output_message}
</synthetic_output_message>
"""
