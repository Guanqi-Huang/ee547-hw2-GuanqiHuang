#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import json
import os
import re
import sys
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_JSON = os.path.join(SCRIPT_DIR, "sample_data", "papers.json")

def now_local(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except FileNotFoundError: return None

class DataStore:
    def __init__(self):
        self.papers_path = PAPERS_JSON
        self.papers = load_json(self.papers_path) or []
        self.by_id = {}
        for p in self.papers:
            aid = p.get("arxiv_id") or p.get("id")
            if isinstance(aid, str): self.by_id[aid] = p
    def exists(self): return bool(self.papers)

DATA = DataStore()
WORD_RE = re.compile(r"[A-Za-z]+")
def tokenize(txt):
    return WORD_RE.findall((txt or "").lower())

def abstract_stats(p):
    text = p.get("abstract", "") or ""
    words = tokenize(text)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "total_sentences": len(sents),
    }

def count_terms(haystack, terms):
    total = 0; in_title = False; in_abs = False
    for t in terms:
        pat = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        if "title" in haystack:
            c = len(pat.findall(haystack["title"])); total += c; in_title |= c > 0
        if "abstract" in haystack:
            c = len(pat.findall(haystack["abstract"])); total += c; in_abs |= c > 0
    where = []
    if in_title: where.append("title")
    if in_abs:   where.append("abstract")
    return total, where

def send_json(h, obj, status=200, log_extra=""):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(payload)))
    h.end_headers()
    h.wfile.write(payload)
    status_msg = {200:"200 OK",400:"400 Bad Request",404:"404 Not Found",500:"500 Internal Server Error"}.get(status,str(status))
    print(f"[{now_local()}] {h.command} {h.path} - {status_msg} {log_extra}", flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if not DATA.exists():
                send_json(self, {"error":"papers.json not found or empty"}, 500); return

            parsed = urlparse(self.path)
            path = parsed.path
            parts = [unquote(x) for x in path.strip("/").split("/") if x]

            # GET /papers
            if path == "/papers" and len(parts) == 1:
                rows = []
                for p in DATA.papers:
                    rows.append({
                        "arxiv_id": p.get("arxiv_id") or p.get("id",""),
                        "title": p.get("title",""),
                        "authors": p.get("authors", []),
                        "categories": p.get("categories", []),
                    })
                send_json(self, rows, 200, f"({len(rows)} results)")
                return

            # GET /papers/{arxiv_id}
            if len(parts) == 2 and parts[0] == "papers":
                aid = parts[1]
                p = DATA.by_id.get(aid)
                if not p:
                    send_json(self, {"error":"unknown paper id","arxiv_id":aid}, 404); return
                out = {
                    "arxiv_id": p.get("arxiv_id") or p.get("id",""),
                    "title": p.get("title",""),
                    "authors": p.get("authors", []),
                    "abstract": p.get("abstract",""),
                    "categories": p.get("categories", []),
                    "published": p.get("published") or p.get("updated") or "",
                    "abstract_stats": abstract_stats(p),
                }
                send_json(self, out, 200, "(1 result)")
                return

            # GET /search?q=...
            if path == "/search" and len(parts) == 1:
                q = (parse_qs(parsed.query).get("q", [""])[0] or "").strip()
                if not q:
                    send_json(self, {"error":"missing query parameter 'q'"}, 400); return
                terms = tokenize(q)
                if not terms:
                    send_json(self, {"error":"malformed query"}, 400); return
                results = []
                for p in DATA.papers:
                    hay = {"title": p.get("title",""), "abstract": p.get("abstract","")}
                    score, where = count_terms(hay, terms)
                    if score > 0:
                        results.append({
                            "arxiv_id": p.get("arxiv_id") or p.get("id",""),
                            "title": p.get("title",""),
                            "match_score": int(score),
                            "matches_in": where,
                        })
                send_json(self, {"query": q, "results": results}, 200, f"({len(results)} results)")
                return

            # GET /stats
            if path == "/stats" and len(parts) == 1:
                total_papers = len(DATA.papers)
                total_words = 0; uniq = set(); freq = {}; cat = {}
                for p in DATA.papers:
                    words = tokenize(p.get("abstract",""))
                    total_words += len(words); uniq.update(words)
                    for w in words: freq[w] = freq.get(w, 0) + 1
                    for c in p.get("categories", []): cat[c] = cat.get(c, 0) + 1
                top_10 = [{"word": w, "frequency": n}
                          for w, n in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:10]]
                out = {
                    "total_papers": total_papers,
                    "total_words": int(total_words),
                    "unique_words": int(len(uniq)),
                    "top_10_words": top_10,
                    "category_distribution": cat,
                }
                send_json(self, out, 200, f"(papers={total_papers})")
                return
            # unknown
            send_json(self, {"error":"endpoint not found"}, 404)
        except Exception as e:
            send_json(self, {"error":"internal server error", "detail": str(e)}, 500)

    def log_message(self, *args, **kwargs):
        return

def main():
    port = 8080
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        try: port = int(sys.argv[1])
        except ValueError:
            print("Error: port must be numeric", file=sys.stderr); sys.exit(1)
    if not (1024 <= port <= 65535):
        print("Error: port must be 1024..65535", file=sys.stderr); sys.exit(1)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Starting ArXiv API server on port {port}")
    print(f"Access at: http://localhost:{port}")
    print("Available endpoints:\n  GET /papers\n  GET /papers/{arxiv_id}\n  GET /search?q=...\n  GET /stats\n")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == "__main__":
    main()
