## Generated using Copilot
## This script converts a Jupyter notebook to a requirements.txt file
import ast
import nbformat
from nbconvert import PythonExporter

def get_imports(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        return [node.module]
    else:
        return []

def notebook_to_requirements(notebook_path, requirements_path):
    # Convert the Jupyter notebook to a Python script
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    python_script, _ = exporter.from_notebook_node(nb)

    # Parse the Python script
    module = ast.parse(python_script)

    # Extract the libraries
    libraries = []
    for node in ast.walk(module):
        libraries.extend(get_imports(node))

    # Write the libraries to the requirements.txt file
    with open(requirements_path, 'w') as f:
        for library in sorted(set(libraries)):
            f.write(f'{library}\n')

# Usage
notebook_to_requirements('your_notebook.ipynb', 'requirements.txt')