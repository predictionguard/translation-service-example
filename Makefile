# ==============================================================================
# Install Commands

install-venv:
	python3 -m venv venv
	source venv/bin/activate && python3 -m pip install -r requirements.txt

install-ui:
	cd frontend/ui && npm install

install-all: install-venv install-ui

# ==============================================================================
# Run Commands

run-backend:
	source venv/bin/activate && python3 main.py

run-ui:
	cd frontend/ui && npm run dev

# ==============================================================================
# Clean Commands

clean-venv:
	rm -rf venv

clean-ui:
	cd frontend/ui && rm -rf node_modules