import json
import time

with open("causal_volatility_framework_60_20_20.ipynb", "r") as f:
    nb = json.load(f)

code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
full_code = ""
for cell in code_cells:
    full_code += "".join(cell) + "\n"

# I will profile the code to see where it hangs
with open("run_all_timed.py", "w") as f:
    f.write("import time\nt_start_global = time.time()\n")
    # Split the code roughly at section comments
    lines = full_code.split('\n')
    for line in lines:
        if line.startswith("# =="):
            f.write("print(f'Time elapsed so far: {time.time() - t_start_global:.2f}s')\n")
        f.write(line + "\n")

