import json
import traceback

with open("causal_volatility_framework.ipynb", "r") as f:
    nb = json.load(f)

code_to_run = ""
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        code_to_run += source + "\n\n"
        if "Section 3 GARCH Residual Filters Applied" in source or "SECTION 3.2:" in source:
            pass
        if "SECTION 4.1:" in source:
            break

try:
    exec(code_to_run)
except Exception as e:
    print("ERROR CAUGHT:")
    traceback.print_exc()
