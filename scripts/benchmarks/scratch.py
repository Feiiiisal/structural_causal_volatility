import json

with open("causal_volatility_framework_60_20_20.ipynb", "r") as f:
    nb = json.load(f)

code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']

# We just want to execute up to Section 3.2
code = ""
for cell in code_cells[:5]:
    code += "".join(cell) + "\n"

with open("run_sec123.py", "w") as f:
    f.write(code)

