window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "none",
    macros: {
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      E: "\\mathbb{E}",
      Var: "\\operatorname{Var}",
      Cov: "\\operatorname{Cov}",
      Ent: "\\mathrm{H}",
      MI: "\\mathrm{I}",
      KL: ["D_{\\mathrm{KL}}\\!\\left(#1\\,\\middle\\|\\,#2\\right)", 2],
      argmax: "\\operatorname*{arg\\,max}",
      argmin: "\\operatorname*{arg\\,min}",
      softmax: "\\operatorname{softmax}",
      diag: "\\operatorname{diag}",
      tr: "\\operatorname{tr}",
      rank: "\\operatorname{rank}",
      indep: "\\perp\\!\\!\\!\\perp",
      do: "\\operatorname{do}",
      Normal: "\\mathcal{N}",
      pa: "\\operatorname{pa}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
