#!/usr/bin/env python3
"""
verify_tables.py

Compares reproduce script outputs against the published table values in main.tex.
Run AFTER executing the reproduce scripts (00.py, 03.py, 05.py) from reproduce/py_scripts/.

Usage (from repo root or cascad_533/):
    python cascad_533/verify_tables.py

For Table 1 to be checked, first capture 00.py stdout:
    cd reproduce/py_scripts
    python 00.py | tee OUTPUT/Table_1_Scalability/table_1_output.txt
"""

import re
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEX_FILE = REPO_ROOT / "main.tex"
OUTPUT_DIR = REPO_ROOT / "reproduce" / "py_scripts" / "OUTPUT"

# ANSI colours (only when stdout is a real terminal)
_tty = sys.stdout.isatty()
GREEN = "\033[32m" if _tty else ""
RED   = "\033[31m" if _tty else ""
YELLOW = "\033[33m" if _tty else ""
RESET = "\033[0m"  if _tty else ""


# -- LaTeX helpers -------------------------------------------------------------

def _strip_tex(s: str) -> str:
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]+)\}", r"\1", s)
    s = s.replace(r"$\pm$", "+-")   # raw string matches literal $\pm$ in file
    return s


def _ints(s: str) -> list:
    return [int(x) for x in re.findall(r"\b\d+\b", s)]


def _floats(s: str) -> list:
    return [float(x) for x in re.findall(r"\d+\.\d+", s)]


# -- LaTeX table parsers -------------------------------------------------------

def _parse_table1_latex(tex: str) -> dict:
    """Returns {scenario_num: {direction: [int, ...]}} from the paths-lenghts table."""
    m = re.search(r"\\label\{tab:paths-lenghts\}(.*?)\\end\{tabular\}", tex, re.DOTALL)
    if not m:
        return {}

    block = _strip_tex(m.group(1))
    result = {1: {}, 2: {}, 3: {}}
    current = None

    for line in block.splitlines():
        if "Scenario 1" in line:
            current = 1
        elif "Scenario 2" in line:
            current = 2
        elif "Scenario 3" in line:
            current = 3
        if current is None:
            continue
        for direction in ("Forward", "Backward", "Difference"):
            if f"quad {direction}" in line:
                # Values are in cells after the first &
                cells = line.split("&")[1:]
                nums = [int(re.search(r"\d+", c).group()) for c in cells if re.search(r"\d+", c)]
                if nums:
                    result[current][direction] = nums[:9]

    return result


def _parse_table3_latex(tex: str) -> dict:
    """Returns {key: (avg, std)} from the threshold_metrics table."""
    m = re.search(r"\\label\{tab:threshold_metrics\}(.*?)\\end\{table\}", tex, re.DOTALL)
    if not m:
        return {}

    block = _strip_tex(m.group(1))
    patterns = [
        ("F1-Score", r"Maximize F1-Score\s*&\s*([\d.]+)\s*&\s*([\d.]+)"),
        ("PRBE",     r"Precision-Recall Break-Even\s*&\s*([\d.]+)\s*&\s*([\d.]+)"),
        ("Youden",   r"Maximize Youden.s J\s*&\s*([\d.]+)\s*&\s*([\d.]+)"),
        ("DROC",     r"Minimize[^&\n]*&\s*([\d.]+)\s*&\s*([\d.]+)"),
    ]
    return {k: (float(hit.group(1)), float(hit.group(2)))
            for k, pat in patterns
            if (hit := re.search(pat, block))}


def _parse_table4_latex(tex: str) -> dict:
    """Returns {method: {metric: (mean, std)}} from the overall_results table."""
    m = re.search(r"\\label\{tab:overall_results\}(.*?)\\end\{table\}", tex, re.DOTALL)
    if not m:
        return {}

    block = _strip_tex(m.group(1))
    metrics = ["Accuracy", "Balanced Accuracy", "F1-Score", "Precision", "Recall"]
    result = {}

    for hit in re.finditer(r"^([\w][\w\-]*)\s*&(.*?)\\\\", block, re.MULTILINE):
        method = hit.group(1).strip()
        cells  = hit.group(2).split("&")
        row = {}
        for metric, cell in zip(metrics, cells):
            nums = _floats(cell)
            if len(nums) == 2:
                row[metric] = (nums[0], nums[1])
        if row:
            result[method] = row

    return result


# -- Output file parsers -------------------------------------------------------

def _parse_table1_output(path: Path) -> dict:
    """Returns {scenario_num: {direction: [int, ...]}} from 00.py stdout file."""
    if not path.exists():
        return {}

    result = {1: {}, 2: {}, 3: {}}
    current = None

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-16")

    for line in content.splitlines():
        if "SCENARIO 1" in line.upper():
            current = 1
        elif "SCENARIO 2" in line.upper():
            current = 2
        elif "SCENARIO 3" in line.upper():
            current = 3
        if current is None:
            continue
        low = line.strip().lower()
        for direction, key in[("forward", "Forward"), ("backward", "Backward"), ("delta", "Difference")]:
            if low.startswith(direction):
                nums = _ints(line)
                if nums:
                    result[current][key] = nums[:9]

    return result


def _parse_table3_output(path: Path) -> dict:
    """Returns {key: (avg, std)} from table_3_output.txt."""
    if not path.exists():
        return {}

    key_map = {
        "Maximize F1-Score":           "F1-Score",
        "Precision-Recall Break-Even": "PRBE",
        "Maximize Youden's J":         "Youden",
        "Minimize D_ROC(0,1)":         "DROC",
    }
    result = {}

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-16")

    for line in content.splitlines():
        for label, key in key_map.items():
            if line.strip().startswith(label):
                nums = _floats(line)
                if len(nums) == 2:
                    result[key] = (nums[0], nums[1])
    return result


def _parse_table4_output(path: Path) -> dict:
    """Returns {method: {metric: (mean, std)}} from overall_macro_summary.csv."""
    if not path.exists():
        return {}

    result = {}
    metrics =["Accuracy", "Balanced Accuracy", "F1-Score", "Precision", "Recall"]

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-16")

    for row in csv.DictReader(content.splitlines()):
        method = row["Method"].strip()
        if method == "D2C":       # CSV uses D2C; paper uses TD2C
            method = "TD2C"
        m_data = {}
        for metric in metrics:
            nums = _floats(row.get(metric, ""))
            if len(nums) == 2:
                m_data[metric] = (nums[0], nums[1])
        if m_data:
            result[method] = m_data
    return result


# -- Comparison helpers --------------------------------------------------------

def _status(ok: bool) -> str:
    if ok:
        return f"{GREEN}PASS{RESET}"
    return f"{RED}FAIL{RESET}"


def _row(label: str, exp, act, atol: float, all_pass: list) -> None:
    if exp is None or act is None:
        print(f"  {label:<42}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}  {YELLOW}SKIP{RESET}")
        return
    delta = abs(exp - act)
    ok = delta <= atol
    if not ok:
        all_pass.append(False)
    print(f"  {label:<42}  {exp:>10.4f}  {act:>10.4f}  {delta:>8.4f}  {_status(ok)}")


# -- Main function -------------------------------------------------------------

def does_table_matches_latex(
    atol_t3: float = 0.015,
    atol_t4: float = 0.0001,
) -> bool:
    """
    Compare reproduce script outputs against main.tex table values.

    Parameters
    ----------
    atol_t3 : tolerance for Table 3 (threshold metrics; varies with numpy/imblearn version)
    atol_t4 : tolerance for Table 4 (overall results; exact match expected from pre-computed pkl)

    Returns True if every checked value is within tolerance.
    """
    if not TEX_FILE.exists():
        print(f"{RED}ERROR:{RESET} main.tex not found at {TEX_FILE}")
        return False

    tex = TEX_FILE.read_text(encoding="utf-8")
    all_pass: list = []   # append False on each failure

    col_header = f"  {'Label':<42}  {'LaTeX':>10}  {'Output':>10}  {'diff':>8}  Result"

    # -- Table 1: path counts (deterministic, integers) ------------------------
    t1_out_path = OUTPUT_DIR / "Table_1_Scalability" / "table_1_output.txt"
    t1_exp = _parse_table1_latex(tex)
    t1_act = _parse_table1_output(t1_out_path)

    print()
    print("=" * 72)
    print("TABLE 1 - Open Information Paths (integer counts, atol=0)")
    print("=" * 72)

    if not t1_exp:
        print(f"  {YELLOW}SKIP{RESET} Could not parse Table 1 from main.tex")
    elif not t1_act:
        print(f"  {YELLOW}SKIP{RESET} No output file found.")
        print(f"       Capture first:  cd reproduce/py_scripts")
        print(f"       python 00.py | tee OUTPUT/Table_1_Scalability/table_1_output.txt")
    else:
        print(col_header)
        for scen_num in (1, 2, 3):
            for direction in ("Forward", "Backward", "Difference"):
                exp_vals = t1_exp.get(scen_num, {}).get(direction)
                act_vals = t1_act.get(scen_num, {}).get(direction)
                if exp_vals is None or act_vals is None:
                    label = f"Scen {scen_num} {direction}"
                    print(f"  {label:<42}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}  {YELLOW}SKIP{RESET}")
                    continue
                ok = exp_vals == act_vals
                if not ok:
                    all_pass.append(False)
                label = f"Scen {scen_num} {direction}"
                exp_str = " ".join(str(v) for v in exp_vals)
                act_str = " ".join(str(v) for v in act_vals)
                status  = _status(ok)
                if ok:
                    print(f"  {label:<42}  {status}   [{exp_str}]")
                else:
                    print(f"  {label:<42}  {status}")
                    print(f"    LaTeX : [{exp_str}]")
                    print(f"    Output: [{act_str}]")

    # -- Table 3: threshold selection metrics ----------------------------------
    t3_exp = _parse_table3_latex(tex)
    t3_act = _parse_table3_output(OUTPUT_DIR / "Table_3_Threshold" / "table_3_output.txt")

    print()
    print("=" * 72)
    print(f"TABLE 3 - Threshold Selection Metrics  (atol={atol_t3})")
    print("  Note: small residual differences are expected due to numpy/imblearn")
    print("  version variance; bootstrap=True locks the imblearn 0.12.4 behaviour.")
    print("=" * 72)

    if not t3_exp:
        print(f"  {YELLOW}SKIP{RESET} Could not parse Table 3 from main.tex")
    elif not t3_act:
        print(f"  {YELLOW}SKIP{RESET} Output file missing - run 03.py first")
    else:
        print(col_header)
        label_map = {
            "F1-Score": "Maximize F1-Score",
            "PRBE":     "Precision-Recall Break-Even",
            "Youden":   "Maximize Youden's J",
            "DROC":     "Minimize D_ROC(0,1)",
        }
        for key in ("F1-Score", "PRBE", "Youden", "DROC"):
            label = label_map[key]
            exp_pair = t3_exp.get(key)
            act_pair = t3_act.get(key)
            for col_name, idx in (("Avg", 0), ("Std", 1)):
                exp = exp_pair[idx] if exp_pair else None
                act = act_pair[idx] if act_pair else None
                _row(f"{label} ({col_name})", exp, act, atol_t3, all_pass)

    # -- Table 4: overall performance on synthetic data ------------------------
    t4_exp = _parse_table4_latex(tex)
    t4_act = _parse_table4_output(OUTPUT_DIR / "Table_4_Synthetic" / "overall_macro_summary.csv")

    print()
    print("=" * 72)
    print(f"TABLE 4 - Overall Performance on Synthetic Data  (atol={atol_t4})")
    print("  Exact match expected when using the pre-computed causal_dfs_TEST.pkl.")
    print(f"  pkl MD5: 4b49870ad8685e2cb3885d3495d1b9a6")
    print("=" * 72)

    methods_order = ["TD2C", "DYNOTEARS", "Granger", "MVGC", "PCMCI", "PCMCI-GPDC", "VAR", "VARLiNGAM"]
    metrics       = ["Accuracy", "Balanced Accuracy", "F1-Score", "Precision", "Recall"]

    if not t4_exp:
        print(f"  {YELLOW}SKIP{RESET} Could not parse Table 4 from main.tex")
    elif not t4_act:
        print(f"  {YELLOW}SKIP{RESET} Output file missing - run 05.py first")
    else:
        print(col_header)
        for method in methods_order:
            exp_row = t4_exp.get(method, {})
            act_row = t4_act.get(method, {})
            for metric in metrics:
                for col_name, idx in (("mean", 0), ("std", 1)):
                    exp = exp_row[metric][idx] if metric in exp_row else None
                    act = act_row[metric][idx] if metric in act_row else None
                    _row(f"{method}  {metric} ({col_name})", exp, act, atol_t4, all_pass)

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 72)
    if not all_pass:
        print(f"RESULT: {GREEN}ALL CHECKS PASSED{RESET}")
    else:
        n_fail = sum(1 for x in all_pass if not x)
        print(f"RESULT: {RED}{n_fail} CHECK(S) FAILED{RESET} - see above")
    print("=" * 72)
    print()

    return len(all_pass) == 0


if __name__ == "__main__":
    ok = does_table_matches_latex()
    sys.exit(0 if ok else 1)
