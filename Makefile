# Variables
PYTHON = python3
PIP = pip3
VENV_DIR = venv
MAIN = main.py

# Install dependencies
install: 
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created."
	@echo "🚀 Activating the virtual environment..."
	. $(VENV_DIR)/bin/activate && $(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed."

# Code verification (Formatting, Quality, Security)
check:
	@echo "🔍 Checking formatting with Black..."
	$(VENV_DIR)/bin/black --check .
	@echo "✅ Formatting OK!"

	# @echo "🔍 Checking code quality with Flake8..."
	# $(VENV_DIR)/bin/flake8 .
	# @echo "✅ Code quality OK!"

	# @echo "🔍 Checking security with Bandit..."
	# $(VENV_DIR)/bin/bandit -r .
	# @echo "✅ Code security OK!"

#  Prepare data
prepare:
	$(VENV_DIR)/bin/python $(MAIN) --prepare

#  Train the model
train:
	$(VENV_DIR)/bin/python $(MAIN) --train

#  Evaluate the model
evaluate:
	$(VENV_DIR)/bin/python $(MAIN) --evaluate

#  Run tests
test:
	$(VENV_DIR)/bin/pytest tests/

# Clean unnecessary files
clean:
	rm -rf $(VENV_DIR)
	rm -f *.joblib *.png
	@echo "🧹 Cleanup complete."

# Help - List available commands
help:
	@echo "📌 Available commands:"
	@echo "  make install    - Create a virtual environment and install dependencies"
	@echo "  make check      - Verify formatting, code quality, and security"
	@echo "  make prepare    - Prepare data for training"
	@echo "  make train      - Train the model"
	@echo "  make evaluate      - Evaluate the model"
	@echo "  make test       - Run unit tests"
	@echo "  make clean      - Remove the virtual environment and temporary files"

# Executes all the pipeline
all: install check prepare train evaluate test