BACKEND_PORT=8000


.PHONY: install
install:
	@echo "Install dependencies"
	poetry install

# Start backend (fastapi uvicorn)
.PHONY: start
start:
	@echo "Start app"
	uvicorn app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT)

