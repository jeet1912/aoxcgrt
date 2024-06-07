#Generated using GPT4o.

import ast
import nbformat
from nbconvert import PythonExporter
import os

def get_imports(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        return [node.module]
    else:
        return []

def get_top_level_package(module_name):
    return module_name.split('.')[0]

def notebook_to_requirements(directory_path, requirements_path):
    libraries = set()

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                with open(notebook_path) as f:
                    nb = nbformat.read(f, as_version=4)
                exporter = PythonExporter()
                python_script, _ = exporter.from_notebook_node(nb)

                module = ast.parse(python_script)

                for node in ast.walk(module):
                    imports = get_imports(node)
                    top_level_imports = [get_top_level_package(lib) for lib in imports]
                    libraries.update(top_level_imports)

    with open(requirements_path, 'w') as f:
        for library in sorted(libraries):
            f.write(f'{library}\n')

# Usage
notebook_to_requirements('code', 'requirements.txt')