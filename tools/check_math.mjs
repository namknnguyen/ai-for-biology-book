// Validate every math expression in the built site with MathJax's TeX parser.
// Usage: NODE_PATH=<dir with node_modules> node tools/check_math.mjs [site_dir]
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const require = createRequire(process.env.MJ_NODE_MODULES + "/");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const siteDir = process.argv[2] || "site";
const cfgSrc = fs.readFileSync("docs/assets/mathjax.js", "utf8");
const win = {};
new Function("window", cfgSrc)(win);
const macros = win.MathJax.tex.macros;

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages.filter((p) => p !== "bussproofs"), macros, formatError: (jax, err) => { throw err; } });
const svg = new SVG({ fontCache: "none" });
const doc = mathjax.document("", { InputJax: tex, OutputJax: svg });

const unescape = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
function walk(d) {
  return fs.readdirSync(d, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(d, e.name)) : e.name.endsWith(".html") ? [path.join(d, e.name)] : []);
}
let total = 0, bad = 0;
for (const f of walk(siteDir)) {
  const html = fs.readFileSync(f, "utf8");
  const re = /<(span|div) class="arithmatex">([\s\S]*?)<\/\1>/g;
  let m;
  while ((m = re.exec(html))) {
    let src = unescape(m[2]).trim();
    let display = false;
    if (src.startsWith("\\(")) src = src.slice(2, -2);
    else if (src.startsWith("\\[")) { src = src.slice(2, -2); display = true; }
    total++;
    try { doc.convert(src, { display }); }
    catch (e) {
      bad++;
      console.log(`MATH ERROR in ${f}: ${String(e.message || e).slice(0, 120)}\n   expr: ${src.slice(0, 160).replace(/\n/g, " ")}`);
    }
  }
}
console.log(`checked ${total} expressions, ${bad} errors`);
process.exit(bad ? 1 : 0);
