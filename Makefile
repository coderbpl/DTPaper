# DarkTrace — convenience targets
# Usage: `make stats`, `make figures`, `make manuscript`, `make paper`
PY ?= python3

.PHONY: stats figures manuscript paper clean help

## stats: run the statistical-validation layer -> stats_results.json + table_stats.csv
stats:
	$(PY) -m src.exp_stats

## figures: regenerate the supplementary result figures
figures:
	$(PY) make_figures.py

## manuscript: build the merged manuscript PDF (pdflatex + bibtex)
manuscript:
	pdflatex -interaction=nonstopmode darktrace_manuscript.tex
	bibtex darktrace_manuscript || true
	pdflatex -interaction=nonstopmode darktrace_manuscript.tex
	pdflatex -interaction=nonstopmode darktrace_manuscript.tex

## paper: alias for manuscript
paper: manuscript

## clean: remove LaTeX build artifacts
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
