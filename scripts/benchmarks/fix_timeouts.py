import json

def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                # Fix yf.download
                if 'auto_adjust=True,' in line:
                    if 'timeout=15' not in source[i+1]:
                        source[i] = line + "        timeout=15,\n"
                
                # Fix pd.read_csv for FRED Mode 2
                if 'df_csv = pd.read_csv(csv_fallback_url' in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    replacement = (
                        indent + "import io\n" +
                        indent + "response = requests.get(csv_fallback_url, timeout=15)\n" +
                        indent + "response.raise_for_status()\n" +
                        indent + "df_csv = pd.read_csv(io.StringIO(response.text), parse_dates=['DATE'], index_col='DATE')\n"
                    )
                    source[i] = replacement
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_notebook('causal_volatility_framework_60_20_20.ipynb')
fix_notebook('causal_volatility_framework_OOS.ipynb')
print("Fixed timeouts in notebooks.")
