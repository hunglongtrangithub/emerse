# EMERSE Summarization

This project builds a Flask server that serves clinical document summarization models on NVIDIA machine with GPU on PORT 5000

# Development

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) for package management

## Installation

1. Clone the repository

```bash
git clone https://github.com/hunglongtrangithub/emerse.git
```

2. Install dependencies

```bash
uv sync
```

## Usage

```bash
uv run run.py
```

## Build with Docker

```bash
make build
```


