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

- [ ] Author name and date (`\author`, `\date`, marked TODO in red).
- [ ] Verify the bibliography entries marked `TODO verify` (FracAtlas author list and article number, MedGemma /
      MedSigLIP arXiv id, Qwen3 and Qwen3-VL references, BiomedCLIP).
- [ ] Adjust the AI-assistance statement to the venue's policy.
- [ ] Every number traces to `research/claims.md` and `results/`; re-check any number you edit.
- [ ] arXiv: first-time submitters in cs.CV / cs.LG / eess.IV need an endorsement.
- [ ] Optional: archive splits, embeddings and the trained heads (e.g. Hugging Face Hub) and cite them.
