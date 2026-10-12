import json

def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if "parse_dates=['DATE']" in line or "index_col='DATE'" in line:
                    source[i] = line.replace("parse_dates=['DATE']", "parse_dates=['observation_date']").replace("index_col='DATE'", "index_col='observation_date'")
                if "df_csv[series_id] = pd.to_numeric(df_csv[series_id], errors='coerce')" in line:
                    # ensure df_csv.index.name = 'DATE' is above this
                    indent = line[:len(line) - len(line.lstrip())]
                    if i > 0 and 'index.name' not in source[i-1]:
                        source.insert(i, indent + "df_csv.index.name = 'DATE'\n")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_notebook('causal_volatility_framework_60_20_20.ipynb')
fix_notebook('causal_volatility_framework_OOS.ipynb')
