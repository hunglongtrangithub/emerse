# Variables
DOCKER_IMAGE_NAME = build_emerse_app
DOCKER_CONTAINER_NAME = build_emerse_app_container
OUTPUT_BINARY_PATH = /app/dist
LOCAL_OUTPUT_PATH = ../test_binary/resources

.PHONY: all
all: build run copy clean lock

.PHONY: lock
lock:
	uv pip freeze | uv pip compile - -o requirements.txt

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

.PHONY: copy
copy:
	@echo "Copying the binary output from Docker container..."
	docker cp $(DOCKER_CONTAINER_NAME):$(OUTPUT_BINARY_PATH) $(LOCAL_OUTPUT_PATH)
	@echo "Binary output copied successfully to $(LOCAL_OUTPUT_PATH)."

.PHONY: clean
clean:
	@echo "Cleaning up Docker container..."
	docker rm -f $(DOCKER_CONTAINER_NAME)
	@echo "Docker container removed successfully."

.PHONY: test
test:
	@echo "Testing the code by running the Docker container..."
	docker run -it --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)
	@echo "Docker container ran successfully for testing."

.PHONY: full-build
full-build: build run copy clean

.PHONY: rebuild-test
rebuild-test: clean build test

