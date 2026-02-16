import re


def extract_verilog_content(text: str) -> str | None:
    """
    Extract Verilog/SystemVerilog code from LLM output using regex.
    Handles:
      ```verilog
      ```systemverilog
      ```sv
      ```
    Falls back to returning None if no fenced block detected.
    """

    # Pattern for all common fenced formats
    pattern = re.compile(r"```(?:verilog|systemverilog|sv)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    matches = pattern.findall(text)

    # 1) If at least one fenced block found → return the FIRST block
    if matches:
        got_match = str(matches[0])
        return got_match.strip()

    # 2) If no fenced code block — common fallback behavior:
    # Try extracting module ... endmodule
    fallback = re.search(r"(module\s+[\s\S]*?endmodule)", text, re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()

    # 3) Absolute fallback → return None
    return None


def extract_filename_from_text(text: str) -> str | None:
    """
    Extract filename from LLM plain-text output using regex patterns.
    Handles:
      - filename: tb.sv
      - file name = tb_top.sv
      - output file is "tb.sv"
      - 'tb.sv' inside common filename statements
    Returns None if no filename found.
    """

    # Common filename patterns
    patterns = [
        r"filename\s*[:=]\s*([A-Za-z0-9_\-./]+\.s?v)",  # filename: tb.sv
        r"file\s*name\s*[:=]\s*([A-Za-z0-9_\-./]+\.s?v)",  # file name = tb.sv
        r"output\s*file\s*(?:is|=)\s*[\"']?([A-Za-z0-9_\-./]+\.s?v)",  # output file is "tb.sv"
        r"\b([A-Za-z0-9_\-]+\.s?v)\b",  # any tb.sv floating alone
    ]

    ret = None

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ret = m.group(1)
            break
    if ret:
        if "/" in ret:
            ret = ret.split("/")[-1]
        if "\\" in ret:
            ret = ret.split("\\")[-1]
    return ret
