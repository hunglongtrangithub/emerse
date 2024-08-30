# EMERSE Summarization
This project builds a binary file executing from python script using PyInstaller, which serves a summarization model on nvidia machine with GPU on PORT 5000

# Development
## Prerequisites
- Python 3.11
- [uv](https://github.com/astral-sh/uv) for package management

## Installation
```bash
uv pip install -r requirements.txt
```

## Usage
```bash
python run.py
```

## Build
```bash
pyinstaller --onefile run.py
```
