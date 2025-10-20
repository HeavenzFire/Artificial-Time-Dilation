PY=python3
VENV=.venv
REQ=requirements.txt
FIGS=figures/d_scaling_chart.png figures/reward_curves.png

.PHONY: help setup figures clean paper web accelerate sweep_plots

help:
	@echo "Targets: setup, figures, paper, web, accelerate, sweep_plots, clean"

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

accelerate:
	$(PY) scripts/dilation_sweep.py --envs 256 --steps 500000 --log_every 10000 --runs_root /workspace/runs --seed 42

sweep_plots:
	$(PY) scripts/plot_acceleration.py --sweep_root /workspace/runs/latest_sweep --outdir /workspace/figures

clean:
	rm -f $(FIGS)
