
.PHONY: start-backend
start-backend:
	@echo "Start backend"
	make -C backend start

.PHONY: start-frontend
start-frontend:
	@echo "Start frontend"
	make -C frontend start

.PHONY: start
start:
	@echo "Start backend and frontend"
	make -j start-backend start-frontend

.PHONY: docker-up
docker-up:
	@echo "Start backend and frontend in docker"
	docker-compose up --build

.PHONY: docker-upd
docker-upd:
	@echo "Start backend and frontend in docker"
	docker-compose up --build -d

.PHONY: docker-down
docker-down:
	@echo "Stop backend and frontend in docker"
	docker-compose down

.PHONY: docker-build
docker-build:
	@echo "Build backend and frontend in docker"
	docker-compose build
