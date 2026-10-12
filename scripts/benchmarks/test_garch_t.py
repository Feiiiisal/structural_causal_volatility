import json

path = 'causal_volatility_framework_60_20_20.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']

# Modify the in-memory code to use dist='t'
for cell in code_cells:
    for i, line in enumerate(cell):
        if "dist='normal'" in line:
            cell[i] = line.replace("dist='normal'", "dist='t'")

code = ""
for cell in code_cells[:5]:
    code += "".join(cell) + "\n"

with open("run_garch_t.py", "w") as f:
    f.write(code)

