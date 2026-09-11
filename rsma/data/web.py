"""Small fetch and text-extraction helpers shared by the corpus builder and the study loop."""
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (rsma; personal research)", "Accept-Encoding": "identity"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding", "").lower()
        if enc == "gzip" or raw[:2] == b"\x1f\x8b":
            import gzip
            raw = gzip.decompress(raw)
        elif enc == "br":
            import brotli
            raw = brotli.decompress(raw)
        return raw.decode("utf-8", errors="ignore")


def strip_html(s):
    s = re.sub(r"<(script|style|nav|header|footer|aside|table|sup)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
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


def feed_entries(url, limit=20):
    """(title, link, summary) from an RSS or Atom feed."""
    root = ET.fromstring(get(url))
    out = []
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title = link = summary = ""
        for c in item:
            t = c.tag.split("}")[-1]
            if t == "title":
                title = (c.text or "").strip()
            elif t == "link":
                link = (c.text or c.attrib.get("href") or "").strip()
            elif t in ("description", "summary", "content"):
                summary = strip_html(c.text or "")
        if link:
            out.append((title, link, summary))
        if len(out) >= limit:
            break
    return out


def wikipedia_search(query, limit=5):
    d = json.loads(get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=%d&srsearch=%s" % (limit, urllib.parse.quote(query))))
    return [r["title"] for r in d["query"]["search"]]


def wikipedia_text(title):
    d = json.loads(get("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles=" + urllib.parse.quote(title)))
    pages = d["query"]["pages"]
    return next(iter(pages.values())).get("extract", "")


def arxiv_search(query, limit=10):
    """(title, id, abstract) for recent arXiv papers matching the query."""
    root = ET.fromstring(get("http://export.arxiv.org/api/query?search_query=%s&sortBy=submittedDate&sortOrder=descending&max_results=%d" % (urllib.parse.quote(query), limit)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        out.append((e.findtext("a:title", "", ns).strip(), e.findtext("a:id", "", ns).strip(),
                    re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()))
    return out
