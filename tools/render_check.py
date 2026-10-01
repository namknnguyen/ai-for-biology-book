"""Render pages of the built site in headless Chromium with MathJax and Mermaid served locally.

Usage: python tools/render_check.py <node_modules_dir> <page.html>... [--shot out.png]
Reports console errors, Mermaid diagrams that failed to render, MathJax errors, and optionally
writes a full-page screenshot. CDN requests are intercepted and answered from node_modules.
"""
import http.server, socketserver, threading, sys, os, functools, json
from playwright.sync_api import sync_playwright

nm = sys.argv[1]
pages = [a for a in sys.argv[2:] if not a.startswith("--")]
shot = None
if "--shot" in sys.argv:
    shot = sys.argv[sys.argv.index("--shot") + 1]
    pages = [p for p in pages if p != shot]

site = os.path.abspath("site")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()

def route(r):
    url = r.request.url
    if "tex-mml-chtml.js" in url:
        r.fulfill(path=os.path.join(nm, "mathjax/es5/tex-mml-chtml.js"), content_type="text/javascript")
    elif "mermaid" in url and url.endswith(".js"):
        r.fulfill(path=os.path.join(nm, "mermaid/dist/mermaid.min.js"), content_type="text/javascript")
    elif url.startswith("http://127.0.0.1"):
        r.continue_()
    elif "fonts.g" in url:
        r.abort()
    else:
        r.abort()

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    for page_path in pages:
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/*", route)
        pg.goto(f"http://127.0.0.1:{port}/{page_path}", wait_until="load")
        pg.wait_for_timeout(2500)
        info = pg.evaluate("""() => ({
            mermaidTotal: document.querySelectorAll('.mermaid').length,
            mermaidSvg: document.querySelectorAll('.mermaid svg').length,
            mermaidErr: [...document.querySelectorAll('.mermaid')].filter(e => /Syntax error|error/i.test(e.textContent) && !e.querySelector('svg g.node')).length,
            mjx: document.querySelectorAll('mjx-container').length,
            mjxErr: document.querySelectorAll('mjx-merror, [data-mjx-error]').length,
            badges: document.querySelectorAll('.ev').length,
            rawTokens: (document.querySelector('.md-content__inner')?.innerText.match(/\\[\\[[ESPHX]\\]\\]/g) || []).length,
            h2: document.querySelectorAll('.md-typeset h2').length,
        })""")
        print(page_path, json.dumps(info), "console errors:", [e[:120] for e in errs][:5])
        if shot:
            pg.screenshot(path=shot, full_page=True)
        pg.close()
    b.close()
httpd.shutdown()
