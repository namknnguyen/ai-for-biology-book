"""Build the single-file PDF of the book.

Pipeline
  1. mkdocs build -f mkdocs-pdf.yml   -> site-pdf/print_page/index.html (every page concatenated)
  2. Headless Chromium loads that page; MathJax and Mermaid are served from a local node_modules
     (so no network is needed), and the script waits until both have finished typesetting.
  3. print CSS is injected (A4, serif body, wrapped code, no clipped tables), a cover page is added,
     and Chromium writes the PDF with page numbers and a bookmark outline.

Usage
  python tools/build_pdf.py <node_modules_dir> [out.pdf] [--sections N] [--no-build]
    --sections N   keep only the first N sections of the print page (quick layout test)
    --no-build     reuse an existing site-pdf/
node_modules_dir must contain mathjax (v3) and mermaid (v11); see notes/PROGRESS.md.
"""
import functools, http.server, json, os, socketserver, subprocess, sys, threading, time
from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith("--")]
nm = os.path.abspath(args[0])
out = os.path.abspath(args[1] if len(args) > 1 else "delivery/ai-for-biology-book.pdf")
nsec = None
if "--sections" in sys.argv:
    nsec = int(sys.argv[sys.argv.index("--sections") + 1])
    args = [a for a in args if a != str(nsec)]
    out = os.path.abspath(args[1] if len(args) > 1 else "delivery/ai-for-biology-book-test.pdf")

if "--no-build" not in sys.argv:
    subprocess.run(["mkdocs", "build", "-f", "mkdocs-pdf.yml", "-q"], check=True)

site = os.path.abspath("site-pdf")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
handler.log_message = lambda *a, **k: None
httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()

MJ_PREFIX = "/mathjax@3/"


def route(r):
    url = r.request.url
    if MJ_PREFIX in url:  # MathJax core and its font files
        rel = url.split(MJ_PREFIX, 1)[1].split("?")[0]
        path = os.path.join(nm, "mathjax", rel)
        if os.path.exists(path):
            ctype = "text/javascript" if path.endswith(".js") else None
            r.fulfill(path=path, content_type=ctype) if ctype else r.fulfill(path=path)
            return
        r.abort()
    elif "mermaid" in url and url.endswith(".js"):
        r.fulfill(path=os.path.join(nm, "mermaid/dist/mermaid.min.js"), content_type="text/javascript")
    elif url.startswith("http://127.0.0.1"):
        r.continue_()
    else:
        r.abort()  # fonts.googleapis etc.: use local fonts


CSS = r"""
@page { size: A4; margin: 20mm 17mm 22mm 17mm;
  @top-left { content: none } @top-center { content: none } @top-right { content: none }
  @bottom-left { content: none } @bottom-center { content: none } @bottom-right { content: none } }
html, body { background: #fff !important; color: #111 !important; }
body { font-family: "Bitstream Charter", "Liberation Serif", "DejaVu Serif", serif !important; }
.md-header, .md-tabs, .md-sidebar, .md-footer, .md-top, .md-search, .md-source, .md-announce,
#print-site-banner, .headerlink, a.headerlink { display: none !important; }
.md-main__inner, .md-content, .md-content__inner, .md-grid { margin: 0 !important; padding: 0 !important; max-width: none !important; }
.md-typeset { font-size: 10.2pt !important; line-height: 1.42 !important; color: #111; font-family: inherit !important; }
.md-typeset p, .md-typeset li { orphans: 3; widows: 3; text-align: left; }
.md-typeset h1 { font-size: 22pt !important; margin: 0 0 8mm !important; line-height: 1.15; color: #0b3d3a !important; }
.md-typeset h2 { color: #1a1a1a !important; }
.md-typeset h3, .md-typeset h4 { color: #222 !important; }
section.print-page.divider { padding-top: 95mm; text-align: center; }
section.print-page.divider h1 { font-size: 30pt !important; color: #00695c !important; }
#print-page-toc ul { list-style: none; margin: 0 !important; padding-left: 5mm !important; }
#print-page-toc > nav > ul { padding-left: 0 !important; }
#print-page-toc li { margin: 0.6mm 0 !important; }
#print-page-toc a { display: flex; align-items: baseline; font-size: 9.6pt; }
#print-page-toc a .pn { margin-left: auto; padding-left: 3mm; color: #333; font-variant-numeric: tabular-nums; }
#print-page-toc a .lead { flex: 1 1 auto; border-bottom: 0.4pt dotted #999; margin: 0 1.5mm; transform: translateY(-1.2mm); }
.md-typeset h2 { font-size: 14.5pt !important; margin-top: 7mm !important; break-after: avoid; }
.md-typeset h3 { font-size: 12pt !important; margin-top: 5mm !important; break-after: avoid; }
.md-typeset h4 { font-size: 10.6pt !important; break-after: avoid; }
section.print-page { break-before: page; }
section.print-page > h1:first-child { break-before: auto; }
.md-typeset code, .md-typeset pre, .md-typeset kbd {
  font-family: "DejaVu Sans Mono", "Liberation Mono", monospace !important; font-size: 7.6pt !important; }
.md-typeset pre { white-space: pre-wrap !important; overflow-wrap: anywhere; word-break: break-word;
  background: #f5f5f5 !important; border: 0.3pt solid #ccc; border-radius: 1.5mm; padding: 2mm 2.5mm; }
.md-typeset pre > code { white-space: pre-wrap !important; overflow: visible !important; padding: 0 !important;
  background: transparent !important; box-shadow: none !important; }
.md-typeset .highlight, .md-typeset .highlighttable { overflow: visible !important; }
.md-typeset .highlight span.filename { font-size: 7.6pt; }
.md-typeset table:not([class]) { font-size: 8.4pt !important; width: 100%; display: table !important;
  table-layout: auto; overflow: visible !important; }
.md-typeset__scrollwrap, .md-typeset__table { overflow: visible !important; display: block; }
.md-typeset table:not([class]) th, .md-typeset table:not([class]) td { overflow-wrap: anywhere; padding: 1mm 1.6mm !important; }
.md-typeset tr { break-inside: avoid; }
.md-typeset .admonition, .md-typeset details { font-size: 9.6pt !important; break-inside: auto; margin: 3.5mm 0 !important;
  box-shadow: none !important; }
.md-typeset .admonition-title, .md-typeset summary { break-after: avoid; }
.md-typeset .arithmatex, mjx-container[display="true"] { overflow: visible !important; max-width: 100%; }
mjx-container[display="true"] { margin: 2mm 0 !important; }
.mermaid, .mermaid svg { max-width: 100% !important; height: auto !important; }
.mermaid { break-inside: avoid; text-align: center; }
img, svg { max-width: 100%; }
.md-typeset a { color: #00695c !important; text-decoration: none; }
.ev { border: 0.3pt solid #999; border-radius: 2pt; padding: 0 2pt; font-size: 7.6pt; }
/* cover */
#pdf-cover { break-after: page; text-align: center; padding-top: 70mm; }
#pdf-cover .kicker { font-size: 11pt; letter-spacing: 0.25em; text-transform: uppercase; color: #00695c; }
#pdf-cover .title { font-size: 40pt; font-weight: 700; color: #0b3d3a; line-height: 1.12; margin: 9mm 0 6mm; }
#pdf-cover .sub { font-size: 14pt; color: #333; max-width: 140mm; margin: 0 auto 18mm; line-height: 1.4; }
#pdf-cover .meta { font-size: 10pt; color: #555; line-height: 1.7; }
#pdf-cover .rule { width: 40mm; border-top: 1.2pt solid #00695c; margin: 0 auto 10mm; }
"""

COVER = """
<div id="pdf-cover">
  <div class="kicker">A research-training textbook</div>
  <div class="title">AI for Biology</div>
  <div class="rule"></div>
  <div class="sub">From First Principles to the Research Frontier</div>
  <div class="meta">59 chapters &middot; 7 appendices &middot; 57 runnable experiments<br>
  Deep learning &amp; foundation models &middot; genomics &middot; proteins &middot; drug discovery<br>
  single-cell &amp; spatial biology &middot; evolution &middot; multimodal models &middot; neuroscience<br><br>
  Online edition: namknnguyen.github.io/ai-for-biology-book<br>
  Many 2025&ndash;26 results are developer-reported; each carries an evidence grade.</div>
</div>
"""

HEADER = "<div></div>"
FOOTER = (
    '<div style="width:100%;font-size:8px;font-family:serif;color:#555;padding:0 17mm;'
    'display:flex;justify-content:space-between">'
    "<span>AI for Biology: From First Principles to the Research Frontier</span>"
    '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1100, "height": 1400})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/*", route)
    pg.set_default_timeout(0)
    t0 = time.time()
    pg.goto(f"http://127.0.0.1:{port}/print_page/", wait_until="load", timeout=600000)
    pg.add_style_tag(content=CSS)
    if nsec:
        pg.evaluate(
            """(n) => { const s = [...document.querySelectorAll('section.print-page')];
                         s.slice(n).forEach(e => e.remove()); }""", nsec)
    pg.emulate_media(media="print")
    pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
    # title-only sections (Part openers, "Front Matter") become divider pages
    pg.evaluate("""() => document.querySelectorAll('section.print-page[id]').forEach(s => {
        if (s.innerText.trim().length < 90 && !s.querySelector('img,svg,table,pre')) s.classList.add('divider'); })""")
    pg.evaluate("(h) => document.querySelector('.md-content__inner, #print-site-page, body').insertAdjacentHTML('afterbegin', h)", COVER)
    # wait for MathJax to finish typesetting the whole document
    pg.wait_for_function("() => window.MathJax && MathJax.startup && MathJax.startup.promise", timeout=120000)
    pg.evaluate("() => MathJax.startup.promise.then(() => MathJax.typesetPromise ? MathJax.typesetPromise() : 0)")
    print("mathjax done %.0fs" % (time.time() - t0), flush=True)
    # wait for Mermaid (Material renders .mermaid blocks into a closed shadow root, so count hosts only)
    pg.wait_for_timeout(5000)
    # a diagram taller than one page is shrunk to fit (width is the knob: the svg scales proportionally)
    shrunk = pg.evaluate("""() => { const maxH = 225 / 25.4 * 96; let n = 0;
        document.querySelectorAll('.mermaid').forEach(m => { const h = m.getBoundingClientRect().height;
          if (h > maxH) { m.style.width = (100 * maxH / h) + '%'; m.style.margin = '0 auto'; n++; } });
        return n; }""")
    pg.evaluate("() => document.fonts.ready")
    info = pg.evaluate("""() => ({
        sections: document.querySelectorAll('section.print-page').length,
        mermaid: document.querySelectorAll('.mermaid').length,
        mjx: document.querySelectorAll('mjx-container').length,
        mjxErr: document.querySelectorAll('mjx-merror, [data-mjx-error]').length,
        rawBadgeTokens: (document.body.innerText.match(/\\[\\[[ESPHX]\\]\\]/g) || []).length,
    })""")
    print("page:", json.dumps(info), "shrunk diagrams:", shrunk, "console errors:", [e[:100] for e in errs if 'ERR_FAILED' not in e][:5], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pdf_args = dict(path=out, format="A4", print_background=True, display_header_footer=True,
                    header_template=HEADER, footer_template=FOOTER, outline=True, tagged=True,
                    margin={"top": "20mm", "bottom": "22mm", "left": "17mm", "right": "17mm"})
    # pass 1: find the page of every heading that appears in the table of contents
    pg.pdf(**pdf_args)
    print("pass 1 written %.0fs" % (time.time() - t0), flush=True)
    from pypdf import PdfReader
    rd = PdfReader(out)
    top = []
    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                top.append((it.title.strip(), rd.get_destination_page_number(it) + 1))
    walk(rd.outline)
    titles = pg.evaluate("""() => [...document.querySelectorAll('#print-page-toc a[href^="#"]')].map(a => {
        const t = document.getElementById(a.getAttribute('href').slice(1)); const h = !t ? null : (t.matches('h1') ? t : t.querySelector('h1')); return h ? h.innerText.trim() : null; })""")
    pages, ptr = [], 0
    for t in titles:
        hit = None
        if t:
            for j in range(ptr, len(top)):
                if top[j][0] in (t, t + t):  # Chromium doubles h1 text that contains a hidden permalink
                    hit, ptr = top[j][1], j + 1
                    break
            if hit is None:  # headings containing math differ in text: fall back to the first 14 characters
                for j in range(ptr, len(top)):
                    if top[j][0][:14] == t[:14]:
                        hit, ptr = top[j][1], j + 1
                        break
        pages.append(hit)
    missing = sum(1 for x in pages if x is None)
    print("toc entries:", len(titles), "without page number:", missing, "total pages:", len(rd.pages), flush=True)
    pg.evaluate("""(pages) => { document.querySelectorAll('#print-page-toc a[href^="#"]').forEach((a, i) => {
        if (pages[i] == null) return;
        const lead = document.createElement('span'); lead.className = 'lead';
        const pn = document.createElement('span'); pn.className = 'pn'; pn.textContent = pages[i];
        a.append(lead, pn); }); }""", pages)
    # pass 2: same layout plus page numbers in the contents
    pg.pdf(**pdf_args)
    rd = PdfReader(out)
    print("wrote", out, "%.1f MB" % (os.path.getsize(out) / 1e6), len(rd.pages), "pages", "%.0fs" % (time.time() - t0), flush=True)
    b.close()
