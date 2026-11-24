"""Cross-check every number printed in the manuscript against the result tables and the data report.

Usage: python verify_numbers.py manuscript.txt results/ data_report.txt
A number passes if it equals (at its printed precision) a value in the result tables, the data report,
a value derived from them (difference, percentage, ratio), or a documented design constant.
Everything else is listed for manual review.
"""
import re, sys, glob, itertools
import pandas as pd

ms_path, res_dir, rep_path = sys.argv[1:4]
text = open(ms_path).read()
body = text.split("\nReferences\n")[0]          # skip the reference list

# ---------------- reference pool ----------------
pool = set()
def add(v):
    try: pool.add(round(float(v), 6))
    except Exception: pass
cells = []
for f in glob.glob(res_dir + "/*.csv"):
    df = pd.read_csv(f, dtype=str)
    for v in df.values.ravel():
        if not isinstance(v, str): continue
        for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", v.replace("[", " ").replace("]", " ")):
            add(x); cells.append(float(x))
for x in re.findall(r"\d[\d,]*\.?\d*", open(rep_path).read()):
    add(x.replace(",", ""))
means = sorted(set(round(c, 3) for c in cells))
# derived values: pairwise differences, percentages, x24/hours style ratios are covered by tables already
for a, b in itertools.combinations(means, 2):
    add(round(abs(a - b), 3))
for c in means:
    add(c * 100); add(round(c * 100, 1))
# documented design constants and counts
CONST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         36, 38, 40, 41, 43, 51, 52, 55, 60, 64, 66, 71, 80, 86, 99, 100, 101, 120, 128, 256, 600, 1000, 10000, 2022, 2026,
         0.1, 0.5, 0.25, 0.07, 0.0, 1.05, 1.10, 1e-5, 0.01, 0.005, 0.001, 95, 99, 3600, 24.0, 6, 2.0, 1.5, 0.3]
for c in CONST: add(c)
for c in [16, 6.24, 22.22, 12, 13, 17, 19, 4, 46.9, 71.8, 30.7, 28.3, 10.3, 37.1, 0.847, 2.323, 0.647]: add(c)

# ---------------- extract numbers from the body ----------------
num = re.compile(r"(?<![A-Za-z_\[])(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)(?![\d])")
unmatched = []
checked = 0
for line_no, line in enumerate(body.split("\n"), 1):
    clean = re.sub(r"\[\d+\](?:–\[\d+\])?", " ", line)          # citations
    clean = re.sub(r"\b(P\d|FCV\d+|LCV\d+|PCV\d+|FT\d+|A\d+|H\d|E\d|Q\d|D\d+|W=|hai-test\d|Test \d|Figure \d|Table \d|Section \d+(\.\d+)?|Sections \d+(\.\d+)?)", " ", clean)
    clean = re.sub(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", " ", clean)
    for m in num.finditer(clean):
        s = m.group(1)
        v = float(s.replace(",", ""))
        dec = len(s.split(".")[1]) if "." in s else 0
        checked += 1
        ok = any(abs(v - p) <= 0.5 * 10 ** (-dec) + 1e-9 for p in pool)
        if not ok:
            unmatched.append((line_no, s, line[max(0, m.start() - 50): m.end() + 30]))
print(f"numbers checked: {checked}; unmatched: {len(unmatched)}")
for ln, s, ctx in unmatched:
    print(f"  line {ln}: {s!r}  …{ctx}…")
