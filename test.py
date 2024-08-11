import os
import glob
import json
from pathlib import Path
from bs4 import BeautifulSoup
if os.getenv("DEBUG"):
    print("Debug mode enabled")
else:
    print("Debug mode disabled")

def test_path():
    path = Path("./input")
    if path.exists():
        print("Path exists")
    filepaths = path.glob('*.json')
    print(list(filepaths))

def get_json_files(directory):
    return glob.glob(os.path.join(directory, '*.json'))

html = """
    <head>
        <title>Test</title>
    </head>
    <body>
        <h1>Test</h1>
        <p>Test paragraph</p>
        <p>Test paragraph</p>
        <p>Test paragraph</p>
    </body>
"""
soup = BeautifulSoup(html, "html.parser")
print(soup.body)
