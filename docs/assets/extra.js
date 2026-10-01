// Replace evidence-grade tokens such as [[E]] with styled badges.
//   [[E]] Established result      [[S]] Strong empirical evidence
//   [[P]] Plausible interpretation [[H]] Open hypothesis   [[X]] Speculative direction
(function () {
  var LABELS = {
    E: ["Established", "Established result: replicated, well understood, or mathematically proven"],
    S: ["Strong evidence", "Strong empirical evidence: multiple independent sources, but not settled"],
    P: ["Plausible", "Plausible interpretation: consistent with the evidence, not yet discriminated from alternatives"],
    H: ["Open hypothesis", "Open hypothesis: a specific claim that experiments could test"],
    X: ["Speculative", "Speculative research direction: motivated, but unproven and possibly wrong"]
  };
  var TOKEN = /\[\[([ESPHX])\]\]/g;
  var SKIP = { PRE: 1, CODE: 1, SCRIPT: 1, STYLE: 1, TEXTAREA: 1 };

  function process(root) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        for (var p = n.parentNode; p && p !== root; p = p.parentNode) {
          if (SKIP[p.nodeName]) return NodeFilter.FILTER_REJECT;
          if (p.classList && p.classList.contains("arithmatex")) return NodeFilter.FILTER_REJECT;
        }
        return TOKEN.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (node) {
      var frag = document.createDocumentFragment();
      var text = node.nodeValue, last = 0, m;
      TOKEN.lastIndex = 0;
      while ((m = TOKEN.exec(text)) !== null) {
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        var span = document.createElement("span");
        span.className = "ev ev-" + m[1];
        span.textContent = LABELS[m[1]][0];
        span.title = LABELS[m[1]][1];
        frag.appendChild(span);
        last = m.index + m[0].length;
      }
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }

  function run() {
    var root = document.querySelector(".md-content__inner");
    if (root) process(root);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
  else run();
})();
