"""Repository checks, run by `make check` and by the check workflow on every push.
 1. Clean tree: no Greek words, no remarks in guillemets, no local absolute paths or host names in any text file.
 2. Page budget: with --pdf, the main text must end within 8 pages. The competition template allows at most 8 pages "excluding
    references and appendices", so everything before the 'References' heading counts, including any statement placed there (the
    statement on the use of large language models therefore stands at the head of the appendix). The page holding that heading is
    the last page of the main text, unless the heading opens the page.
usage: python tools/check_repo.py [--pdf paper/main.pdf]"""
import os, re, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = {".py", ".md", ".sh", ".json", ".csv", ".txt", ".tex", ".bib", ".jobs", ".yaml", ".yml", ".cff"}
BAD = re.compile("[\u0370-\u03FF\u1F00-\u1FFF]{2,}|\u00ab|\u00bb|" + "/ho" + "me/|" + "/me" + "dia/|" + "pana" + "giota", re.I)   # escapes, so this file passes its own check
fails = []
for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d != ".git"]
    for fn in files:
        if os.path.splitext(fn)[1].lower() not in TEXT: continue
        p = os.path.join(root, fn)
        for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
            if BAD.search(line): fails.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:100]}")
if "--pdf" in sys.argv:
    pdf = sys.argv[sys.argv.index("--pdf") + 1]
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    last = None; BACK = ("References",)
    for pg in range(1, n + 1):
        lines = [l.strip() for l in subprocess.run(["pdftotext", "-f", str(pg), "-l", str(pg), pdf, "-"], capture_output=True, text=True).stdout.splitlines() if l.strip()]
        hits = [i for i, l in enumerate(lines) if l in BACK]
        if hits:
            last = pg - 1 if hits[0] == 0 else pg; print(f"main text ends at '{lines[hits[0]]}' on page {pg}"); break
    if last is None: fails.append("page budget: no 'References' heading found")
    elif last > 8: fails.append(f"page budget: the main text ends on page {last} (limit 8)")
    else: print(f"page budget OK: the main text ends on page {last}")
for f in fails: print("FAIL", f)
print("checks passed" if not fails else f"{len(fails)} problem(s)")
sys.exit(1 if fails else 0)
