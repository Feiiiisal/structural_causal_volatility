import json

with open("causal_volatility_framework.ipynb", "r") as f:
    nb = json.load(f)

with open("temp_source.py", "w") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        out.write(f"# CELL {i} ({cell.get('cell_type')})\n")
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        out.write(source + "\n\n")
