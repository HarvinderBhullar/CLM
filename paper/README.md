# Paper draft

`main.tex` + `references.bib`; figures in `figures/` are PDF conversions of `assets/*.svg`
(regenerate with `python tools/fracatlas_figures.py`, then
`rsvg-convert -f pdf assets/<name>.svg -o paper/figures/<name>.pdf`).

## Build

```bash
brew install tectonic          # small TeX engine; fetches packages on demand
cd paper && tectonic main.tex  # -> main.pdf
```

or with a full TeX distribution: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

For arXiv, upload `main.tex`, `references.bib` (or the generated `main.bbl`) and `figures/`.

## Before submission

- [x] Author: Harvinder Singh, Independent researcher, harvinderbhullar@gmail.com; date = `\today` (arXiv stamps its own).
- [x] Bibliography identifiers verified (FracAtlas DOI, MedGemma 2507.05201, Qwen3 2505.09388, Qwen3-VL
      2511.21631, BiomedCLIP 2303.00915).
- [ ] Check the AI-assistance statement against the venue's policy (current text is a plain disclosure).
- [ ] Every number traces to `research/claims.md` and `results/`; re-check any number you edit.
- [ ] arXiv: first-time submitters in cs.CV / cs.LG / eess.IV need an endorsement.
- [ ] Optional: archive splits, embeddings and the trained heads (e.g. Hugging Face Hub) and cite them.
