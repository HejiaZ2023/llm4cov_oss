# Example command:
```text
uv run -m llm4cov.eda.worker.get_cov_result \
    --workdir ./tests/resources \
    --tb tb_counter.sv \
    --sv_files counter.sv \
    --dut counter \
    --cov_type all \
    --metrics code toggle block \
    --timeout 600 \
    --outfile result.json
```

# Arguments explanation
```text
uv run -m llm4cov_eda.get_cov_result -h
usage: get_cov_result.py [-h] --workdir WORKDIR --sv_files SV_FILES [SV_FILES ...] --tb TB --dut DUT
                         [--cov_type COV_TYPE] [--metrics [METRICS ...]] [--timeout TIMEOUT] [--outfile OUTFILE]
                         [--skip_detail]

Run xrun + imc and output JSON summary.

options:
  -h, --help            show this help message and exit
  --workdir WORKDIR     Working directory on server
  --sv_files SV_FILES [SV_FILES ...]
                        List of SV source files
  --tb TB               Testbench file name
  --dut DUT             Top DUT name
  --cov_type COV_TYPE   Coverage type (default: all)
  --metrics [METRICS ...]
                        IMC metrics list
  --timeout TIMEOUT     Timeout per stage (sec)
  --outfile OUTFILE     Where to save JSON result
  --skip_detail         Skip IMC detail coverage report
```

# Output Schema
```json
{
  "type": "object",
  "properties": {
    "returncode": { "type": "integer", "description": "Shell exit code of the simulator (0 for success)." },
    "status": { "type": "string", "enum": ["success", "xrun_failed", "imc_merge_failed", "imc_report_summary_failed", "exception"] },
    "err_msg": { "type": "string", "description": "Stderr output or custom error trace if execution fails, includes xrun_fail log" },
    "cov_info": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Module or instance name." },
                "Overall Average": { "type": ["number", "null"] },
                "Overall Covered": { "type": ["number", "null"] },
                "Code Average": { "type": ["number", "null"] },
                "Code Covered": { "type": ["number", "null"] },
                "Fsm Average": { "type": ["number", "null"] },
                "Fsm Covered": { "type": ["number", "null"] },
                "Functional Average": { "type": ["number", "null"] },
                "Functional Covered": { "type": ["number", "null"] },
                "Toggle": { "type": ["number", "null"] },
                "Block": { "type": ["number", "null"] },
                "Assertion": { "type": ["number", "null"] },
            }
          }
        },
        "detail": { "type": "string", "description": "Raw or filtered coverage report text for LLM feedback." },
        "runtime_sec": { "type": "number", "description": "Total simulation wall-clock time." }
      }
    }
  },
  "required": ["returncode", "status", "err_msg"]
}
```
# Example Output
```json
{
  "returncode": 0,
  "status": "success",
  "err_msg": "",
  "cov_info": {
    "summary": [
      {
        "name": "binary_to_bcd",
        "level": 0,
        "Overall Average": 0.9083,
        "Overall Covered": 0.9037999999999999,
        "Code Average": 0.9083,
        "Code Covered": 0.9037999999999999,
        "Fsm Average": null,
        "Fsm Covered": null,
        "Functional Average": null,
        "Functional Covered": null,
        "Toggle": 0.9,
        "Block": 0.9167000000000001,
        "Assertion": null
      }
    ],
    "detail": "[REDACTED]",
    "runtime_sec": 4.286
    }
}
```

# Targeted Share
Source code may be shared with users who holds Cadence Academic License with non-redistribution agreement. Contact maintainers for more details.