# EMERSE Summarization
This project builds a binary file executing from python script using PyInstaller, which serves a summarization model on nvidia machine with GPU on PORT 5000

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
python run.py
```

## Build
```bash
pyinstaller --onefile run.py
```
