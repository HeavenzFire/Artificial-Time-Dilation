PY=python3
VENV=.venv
REQ=requirements.txt
FIGS=figures/d_scaling_chart.png figures/reward_curves.png

.PHONY: help setup figures clean paper web

help:
	@echo "Targets: setup, figures, paper, web, clean"

setup:
	$(PY) -m venv $(VENV) || true
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r $(REQ)

figures: $(FIGS)

figures/d_scaling_chart.png figures/reward_curves.png: scripts/generate_figures.py $(REQ)
	$(PY) scripts/generate_figures.py

paper: paper/figures.tex figures
	@echo "Compile LaTeX with your local latexmk or TeX distribution."
	@echo "Example: cd paper && latexmk -pdf figures.tex"

web:
	@echo "Open web/index.html in a browser (no build step)."

clean:
	rm -f $(FIGS)
