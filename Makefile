# Credit Risk Classification ML Pipeline
# Makefile for build automation

.PHONY: setup train eval clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup  - Create virtual environment and install dependencies"
	@echo "  train  - Execute training pipeline"
	@echo "  eval   - Run evaluation and generate reports"
	@echo "  clean  - Remove generated artifacts"
	@echo "  help   - Show this help message"

# Create virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python -m venv venv
	@echo "Activating virtual environment and installing dependencies..."
ifeq ($(OS),Windows_NT)
	venv\Scripts\activate && pip install --upgrade pip && pip install -r requirements.txt
else
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
endif
	@echo "Setup complete!"

# Execute training pipeline
train:
	@echo "Starting training pipeline..."
	python -m src.train
	@echo "Training complete!"

# Run evaluation and generate reports
eval:
	@echo "Starting evaluation..."
	python -m src.eval
	@echo "Evaluation complete! Check reports/ directory for results."

# Remove generated artifacts
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/
	rm -rf reports/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	@echo "Cleanup complete!"

# Create necessary directories
dirs:
	mkdir -p artifacts reports
