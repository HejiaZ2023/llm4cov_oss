from llm4cov.datasets.types import LlmGenTbContext

PROMPT_TO_BE_INJECTED = """
<generation_rules>
Follow the expert-written rules below:
1. Do not use based literals with expressions
    — illegal: pixel = 8'd(i+1); legal: pixel = i+1 (sized by LHS)
2. When generating SystemVerilog testbenches,
    use only valid hex digits (0-9, A-F) in literals.
    (like 16'hEFGH is invalid because G and H are not valid hex digits).
3. hex literal should starts with number of bits, followed by 'h', then hex digits.
    Each hex digit represents 4 bits, so make sure the total number of bits can hold the hex value.
    (like 4'h1234 is invalid because 4 bits cannot hold 1234; instead, should be 16'h1234).
    and declare all procedurally driven signals (clk, rst, inputs) as logic, not wire.
4. Always define SystemVerilog tasks wrapped by starting word 'task',
    and ending word 'endtask', not 'end'.
5. $monitor is forbidden unless explicitly requested;
    default to $display inside initial or sequential always blocks.
6. When using # delays with arithmetic, always parenthesize the full delay expression:
    #(N * CLK_PERIOD) not #N * CLK_PERIOD.
7. Single driver rule:
    If a signal is connected to a module output or inout port,
    it must be treated as read-only in the testbench,
    and must not appear on the left-hand side of any procedural assignment.
8. DO NOT index $random or $urandom directly (like $urandom[0]);
    assign to a variable or mask/cast the result before bit selection.
9. All module ports must be declared in the module header (ANSI style).
10. Do not declare variables inside for, if, or after statements in procedural blocks;
    declare all variables at the top of the block.
11. Never use variable indices in [msb:lsb] part-selects;
    use indexed part-selects (base +: width or base -: width) instead.
12. If the RTL has a timescale like '`timescale 1ns / 1ps',
    repeat it at the beginning of the testbench file.

</generation_rules>
"""


def inject_prompt_into_tb_generation(context: LlmGenTbContext) -> LlmGenTbContext:
    context.instructions += "\n" + PROMPT_TO_BE_INJECTED
    return context
