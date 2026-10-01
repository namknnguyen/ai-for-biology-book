"""Make every standalone `$$` display-math fence a proper block for pymdownx.arithmatex.

Python-Markdown only recognizes a `$$ ... $$` block when it is separated from neighbouring
paragraphs by blank lines; a fence glued to the preceding text line is swallowed by the paragraph
and shown as raw TeX. This script inserts the missing blank lines (outside code fences).
Usage: python tools/fix_display_math.py [--check]
"""
import glob, re, sys

check = "--check" in sys.argv
changed = 0
for path in sorted(glob.glob("docs/**/*.md", recursive=True)):
    lines = open(path, encoding="utf-8").read().split("\n")
    out, in_code, in_math = [], False, False
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^(```|~~~)", s):
            in_code = not in_code
        if not in_code and s == "$$":
            if not in_math:  # opening fence: need a blank line before it
                if out and out[-1].strip() != "":
                    out.append("")
                in_math = True
                out.append(line)
                continue
            in_math = False  # closing fence: need a blank line after it
            out.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                out.append("")
            continue
        out.append(line)
    if out != lines:
        changed += 1
        print(("would change " if check else "fixed ") + path, len(out) - len(lines), "lines added")
        if not check:
            open(path, "w", encoding="utf-8").write("\n".join(out))
print(changed, "files")
