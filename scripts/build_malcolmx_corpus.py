"""
Build data_cache/malcolmx.txt from "Malcolm X: Collected Speeches, Debates and Interviews
(1960-1965)", edited by Sandeep S. Atwal. Downloads the PDF, extracts text, repairs ligatures,
unwraps hard line breaks, and normalizes to ASCII.

  .venv/bin/python scripts/build_malcolmx_corpus.py
"""
import os
import re
import sys
import unicodedata
import urllib.request

PDF_URL = "https://ouleft.org/wp-content/uploads/malcolm-collected.pdf"
CACHE = "data_cache"
PDF = os.path.join(CACHE, "malcolm-collected.pdf")
OUT = os.path.join(CACHE, "malcolmx.txt")


def extract(pdf_path):
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from pypdf import PdfReader
    r = PdfReader(pdf_path)
    return "\n".join((p.extract_text() or "") for p in r.pages)


def clean(txt):
    txt = re.sub(r"/([A-Za-z])_", r"\1", txt)                      # /T_he -> The, /f_irst -> first
    for a, b in {"‘": "'", "’": "'", "“": '"', "”": '"', "—": " - ",
                 "–": "-", "…": "...", " ": " "}.items():
        txt = txt.replace(a, b)
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode("ascii")
    lines = txt.split("\n")
    out, buf = [], ""
    for ln in lines:
        if re.fullmatch(r"\s*\d{1,4}\s*", ln):                        # page numbers
            continue
        stripped = ln.rstrip()
        if not stripped:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")
            continue
        if ln.endswith(" ") and not stripped.endswith((".", "!", "?", ":", '"')):  # wrapped line
            buf += stripped + " "
        elif ln.endswith(" "):
            buf += stripped + " "
        else:
            buf += stripped
            out.append(buf.strip())
            buf = ""
    if buf:
        out.append(buf.strip())
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(PDF):
        print("downloading", PDF_URL)
        urllib.request.urlretrieve(PDF_URL, PDF)
    raw = extract(PDF)
    text = clean(raw)
    # drop the front matter: start at the first speech heading after the table of contents
    m = list(re.finditer(r"Harlem Freedom Rally", text))
    if len(m) >= 2:
        text = text[m[1].start():]
    open(OUT, "w").write(text)
    chars = sorted(set(text))
    print(f"wrote {OUT}: {len(text):,} chars, vocab {len(chars)}: {''.join(chars)!r}")


if __name__ == "__main__":
    main()
