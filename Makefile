# Variables
DOCKER_IMAGE_NAME = build_emerse_app
DOCKER_CONTAINER_NAME = build_emerse_app_container

.PHONY: all
all: build run copy clean lock dev format

.PHONY: format
format:
	uv run ruff format .

.PHONY: dev
dev:
	hupper -m run

.PHONY: build
build:
	@echo "Building the Docker image..."
	docker build -t $(DOCKER_IMAGE_NAME) .
	@echo "Docker image built successfully."

.PHONY: run
run:
	@echo "Running the Docker container..."
	docker run -d --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)
	@echo "Docker container started successfully."


.PHONY: clean
clean:
	@echo "Cleaning up Docker container..."
	docker rm -f $(DOCKER_CONTAINER_NAME)
	@echo "Docker container removed successfully."

.PHONY: test
test:
	docker run -it --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)

