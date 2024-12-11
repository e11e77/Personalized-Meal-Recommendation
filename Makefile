PYTHON := $(shell command -v python3 || command -v python)
ifeq ($(PYTHON),)
    $(error "No Python interpreter found.")
endif

PIP := $(shell $(PYTHON) -m pip --version > /dev/null 2>&1 && echo "pip" || echo "pip3")

SCRIPT := src/postprocessing.py

SRC_DIR := src 
DATA_DIR := data 
TEST_DIR := src/tests

install:
	@echo "Installing all the necessary libraries..."
	$(PIP) install -r requirements.txt

run:
	@echo "Running the tests..."
	pytest $(TEST_DIR)
	@echo "Executing the end-to-end pipeline..."
	$(PYTHON) $(SCRIPT)

test:
	@echo "Running the tests..."
	pytest $(TEST_DIR)
