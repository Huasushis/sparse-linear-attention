window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true
  },
  options: { ignoreHtmlClass: "^.*|", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => {
  MathJax.typesetPromise();
});
