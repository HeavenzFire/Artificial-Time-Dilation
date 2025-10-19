# Makefile for Artificial Time Dilation RL project

.PHONY: help install test clean run-demo run-web generate-paper setup

# Default target
help:
	@echo "Artificial Time Dilation for RL - Available Commands"
	@echo "=================================================="
	@echo "setup          - Set up the project environment"
	@echo "install        - Install dependencies"
	@echo "test           - Run test suite"
	@echo "clean          - Clean build artifacts"
	@echo "run-demo       - Run basic time dilation demo"
	@echo "run-web        - Launch web interface"
	@echo "generate-paper - Generate research paper PDF"
	@echo "lint           - Run code linting"
	@echo "format         - Format code with black"

# Setup project
setup:
	@echo "Setting up project environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -m src.cli config --output config.yaml
	@echo "Setup complete!"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	@echo "Running test suite..."
	python tests/test_runner.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Run basic demo
run-demo:
	@echo "Running basic time dilation demo..."
	python examples/basic_time_dilation.py

# Run web interface
run-web:
	@echo "Launching web interface..."
	streamlit run demos/web_demo.py

# Generate research paper
generate-paper:
	@echo "Generating research paper..."
	cd docs/paper && latexmk -pdf main.tex
	@echo "Paper generated: docs/paper/main.pdf"

# Run linting
lint:
	@echo "Running code linting..."
	flake8 src/ tests/ examples/ demos/
	mypy src/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ examples/ demos/

# Run experiment
experiment:
	@echo "Running time dilation experiment..."
	python -m src.cli experiment --env CartPole-v1 --factors 1 10 100 1000 --episodes 10

# Generate visualizations
visualize:
	@echo "Generating visualizations..."
	python -m src.cli visualize --input results/experiment_results.json --output charts/

# Full pipeline
pipeline: setup test experiment visualize generate-paper
	@echo "Full pipeline completed!"

# Development setup
dev-setup: setup
	@echo "Setting up development environment..."
	pip install -e .
	pre-commit install
	@echo "Development setup complete!"