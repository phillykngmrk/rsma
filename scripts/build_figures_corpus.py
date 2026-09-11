"""
Build data_cache/figures/<slug>.txt for every figure in figures.json, from:
  gutenberg / gutenberg_search   Project Gutenberg books (public domain)
  wikisource_author              works linked from a Wikisource author page (opinions, speeches)
  blackpast                      speeches on blackpast.org found by searching the name
  youtube                        transcripts of talks and interviews found by search queries
  urls                           plain-text or HTML pages
  local                          a file already on disk
Writes coverage.json with characters per figure and domain. Everything is cached, rerun is cheap.

  .venv/bin/python scripts/build_figures_corpus.py [--max-videos 10] [--only "Name,Name"]
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache")
OUT = os.path.join(CACHE, "figures")
RAW = os.path.join(CACHE, "figures_raw")
UA = {"User-Agent": "Mozilla/5.0 (rsma corpus builder; personal research)"}
YT_SLEEP, SKIP_YT = 1.0, False
YT_BLOCKED = 0


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip" or raw[:2] == b"\x1f\x8b":
            import gzip
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="ignore")


def cached(key, fn):
    os.makedirs(RAW, exist_ok=True)
    p = os.path.join(RAW, key + ".txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    try:
        t = fn()
    except Exception as e:
        print(f"    ! {key}: {type(e).__name__}: {str(e)[:100]}")
        return ""  # not cached, so a rerun retries it
    open(p, "w", encoding="utf-8").write(t or "")
    return t or ""


def strip_html(s):
    s = re.sub(r"<(script|style|sup|table)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    ps = re.findall(r"<p[^>]*>(.*?)</p>", s, re.S | re.I)
    if ps:
        s = "\n\n".join(ps)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def unwrap(t):
    t = t.replace("\r\n", "\n")
    paras = re.split(r"\n\s*\n", t)
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p).strip() for p in paras if p.strip())


# ---- sources ------------------------------------------------------------------------------
def gutenberg(gid):
    def fetch():
        t = get(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt")
        m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
        if m:
            t = t[m.end():]
        m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
        if m:
            t = t[:m.start()]
        return unwrap(t)
    return cached(f"gutenberg_{gid}", fetch)


def gutenberg_search(q, limit=6):
    def fetch():
        d = json.loads(get(f"https://gutendex.com/books?search={urllib.parse.quote(q)}&languages=en"))
        ids = [b["id"] for b in d["results"][:limit] if any(q.split(",")[0].lower() in a["name"].lower() for a in b["authors"])]
        return json.dumps(ids)
    ids = json.loads(cached(f"gutendex_{slug(q)}", fetch) or "[]")
    return [(f"gutenberg {i}", gutenberg(i)) for i in ids]


def wikisource_page(title):
    def fetch():
        d = json.loads(get("https://en.wikisource.org/w/api.php?action=parse&prop=text&format=json&page=" + urllib.parse.quote(title)))
        return strip_html(d["parse"]["text"]["*"])
    return cached("wikisource_" + slug(title)[:80], fetch)


def wikisource_author(author, limit=80):
    def fetch():
        d = json.loads(get("https://en.wikisource.org/w/api.php?action=parse&prop=links&format=json&page=" + urllib.parse.quote("Author:" + author)))
        return json.dumps([l["*"] for l in d["parse"]["links"] if l["ns"] == 0])
    links = json.loads(cached("wikisource_author_" + slug(author), fetch) or "[]")
    docs = []
    for t in links[:limit]:
        txt = wikisource_page(t)
        if len(txt) > 1500:
            docs.append((f"wikisource {t}", txt))
    return docs


def blackpast(name):
    def search():
        s = get("https://blackpast.org/?s=" + urllib.parse.quote(name))
        urls = re.findall(r'https://blackpast\.org/(?:african-american-history|global-african-history)/[a-z0-9-]*(?:1[789]|20)\d\d[a-z0-9-]*/', s)
        return json.dumps(sorted(set(urls)))
    urls = json.loads(cached("blackpast_search_" + slug(name), search) or "[]")
    last = name.split()[-1].lower()
    docs = []
    for u in urls:
        if last not in u:
            continue
        def fetch(u=u):
            s = get(u)
            m = re.search(r'<div class="entry-content[^"]*"[^>]*>(.*?)</div>\s*<footer', s, re.S)
            body = m.group(1) if m else s
            txt = strip_html(body)
            paras = txt.split("\n\n")
            return "\n\n".join(paras[2:]) if len(paras) > 3 else txt  # drop photo caption and editorial intro
        txt = cached("blackpast_" + slug(u.rstrip("/").split("/")[-1])[:80], fetch)
        if len(txt) > 800:
            docs.append((f"blackpast {u}", txt))
    return docs


def youtube(name, queries, max_videos):
    last = name.split()[-1].lower()
    docs, seen = [], set()
    for q in queries:
        def search(q=q):
            r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "yt-dlp"), f"ytsearch8:{q}", "--flat-playlist",
                                "--print", "%(id)s\t%(duration)s\t%(title)s"], capture_output=True, text=True, timeout=120)
            return r.stdout
        rows = cached("ytsearch_" + slug(q), search).strip().split("\n")
        for row in rows:
            parts = row.split("\t")
            if len(parts) != 3:
                continue
            vid, dur, title = parts
            try:
                dur = float(dur)
            except ValueError:
                dur = 0
            if vid in seen or dur < 420 or last not in title.lower():
                continue
            seen.add(vid)
            if len(seen) > max_videos:
                break
            def fetch(vid=vid):
                from youtube_transcript_api import YouTubeTranscriptApi
                t = YouTubeTranscriptApi().fetch(vid, languages=["en", "en-US", "en-GB"])
                time.sleep(YT_SLEEP)
                txt = " ".join(s.text.replace("\n", " ") for s in t)
                txt = re.sub(r"\[(Music|Applause|Laughter)\]", "", txt, flags=re.I)
                return re.sub(r"\s+", " ", txt).strip()
            global YT_BLOCKED
            if YT_BLOCKED >= 3:
                return docs
            before = YT_BLOCKED
            txt = cached("yt_" + vid, fetch)
            if txt:
                YT_BLOCKED = 0
                if len(txt) > 2000:
                    docs.append((f"youtube {vid} {title}", txt))
            elif not os.path.exists(os.path.join(RAW, "yt_" + vid + ".txt")):
                YT_BLOCKED = before + 1
    return docs


def wikiquote(name):
    """Sourced quotations from Wikiquote, one per line. Short but real words of the person."""
    def fetch():
        d = json.loads(get("https://en.wikiquote.org/w/api.php?action=parse&prop=text&format=json&redirects=1&page=" + urllib.parse.quote(name)))
        body = d["parse"]["text"]["*"]
        # keep the main quotation lists, drop the "About"/"Misattributed"/"Disputed" sections
        for cut in ("Misattributed", "Disputed", "Quotes_about", "About_"):
            i = body.find('id="' + cut)
            if i > 0:
                body = body[:i]
        lis = re.findall(r"<li>(.*?)</li>", body, re.S)
        out = []
        for li in lis:
            t = strip_html(re.sub(r"<ul>.*?</ul>", "", li, flags=re.S)).replace("\n", " ").strip()
            if 40 < len(t) < 2000:
                out.append(t)
        return "\n\n".join(out)
    return cached("wikiquote_" + slug(name), fetch)


def url_text(u):
    def fetch():
        t = get(u)
        if "<html" in t[:2000].lower():
            return strip_html(t)
        m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
        if m:
            t = t[m.end():]
            e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
            t = t[:e.start()] if e else t
        return unwrap(t)
    return cached("url_" + slug(u)[-80:], fetch)


# ---- main ---------------------------------------------------------------------------------
def build(fig, max_videos):
    docs = []
    for gid in fig.get("gutenberg", []):
        docs.append((f"gutenberg {gid}", gutenberg(gid)))
    if fig.get("gutenberg_search"):
        docs += gutenberg_search(fig["gutenberg_search"])
    if fig.get("wikisource_author"):
        docs += wikisource_author(fig["wikisource_author"])
    if fig.get("blackpast"):
        docs += blackpast(fig["name"])
    if fig.get("youtube") and not SKIP_YT:
        docs += youtube(fig["name"], fig["youtube"], max_videos)
    for u in fig.get("urls", []):
        docs.append((f"url {u}", url_text(u)))
    if fig.get("wikiquote", True):
        docs.append(("wikiquote", wikiquote(fig.get("wikiquote_page") or fig["name"])))
    if fig.get("local"):
        docs.append((f"local {fig['local']}", open(os.path.join(ROOT, fig["local"]), encoding="utf-8").read()))
    docs = [(s, t) for s, t in docs if t and len(t) > 500]
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=10)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-youtube", action="store_true", help="public-domain and web sources only")
    ap.add_argument("--yt-sleep", type=float, default=1.0, help="seconds between transcript fetches")
    args = ap.parse_args()
    global YT_SLEEP, SKIP_YT
    YT_SLEEP, SKIP_YT = args.yt_sleep, args.skip_youtube
    manifest = json.load(open(os.path.join(ROOT, "figures.json")))
    os.makedirs(OUT, exist_ok=True)
    only = set(n.strip() for n in args.only.split(",")) if args.only else None
    coverage = {"persona_name": manifest["persona_name"], "figures": {}, "domains": {}}
    for fig in manifest["figures"]:
        if only and fig["name"] not in only:
            continue
        print(f"{fig['name']} ({fig['domain']})")
        docs = build(fig, args.max_videos)
        text = "".join(f"\n\n### {fig['name']} | {src}\n\n{t.strip()}\n" for src, t in docs)
        path = os.path.join(OUT, slug(fig["name"]) + ".txt")
        open(path, "w", encoding="utf-8").write(text)
        n = len(text)
        coverage["figures"][fig["name"]] = {"domain": fig["domain"], "chars": n, "docs": len(docs), "file": os.path.relpath(path, ROOT),
                                            "sources": [s.split(" ")[0] for s, _ in docs]}
        coverage["domains"][fig["domain"]] = coverage["domains"].get(fig["domain"], 0) + n
        print(f"    {len(docs)} docs, {n:,} chars")
        json.dump(coverage, open(os.path.join(OUT, "coverage.json"), "w"), indent=1)
    print("\nby domain:", {k: f"{v/1e6:.2f}M chars" for k, v in sorted(coverage["domains"].items())})


if __name__ == "__main__":
    main()
