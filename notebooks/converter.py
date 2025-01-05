"""Execute all notebooks in the 'notebooks' folder to prepare them for publishing as GitHub pages.
Launch it from the root directory.
"""

import os
import sys

from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write

# Append the parent directory of 'src' to sys.path
src_parent_path = os.path.abspath(os.getcwd())

offline_rl_path = os.path.join(src_parent_path, "src", "offline_rl")
print(f"Appending {offline_rl_path} to sys.path")
sys.path.append(offline_rl_path)

# Debug sys.path
print("sys.path content after appending:")
for path in sys.path:
    print(path)

# Define the notebooks directory
notebooks_dir = os.path.join(src_parent_path, "notebooks")

# Create an ExecutePreprocessor instance
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

# Loop through all notebooks in the notebooks directory
for notebook_filename in os.listdir(notebooks_dir):
    if notebook_filename.endswith(".ipynb"):
        notebook_path = os.path.join(notebooks_dir, notebook_filename)
        print(f"Processing notebook: {notebook_path}")

        # Open the notebook and read it
        with open(notebook_path) as f:
            notebook_content = read(f, as_version=4)

        # Execute the notebook
        try:
            ep.preprocess(notebook_content, {"metadata": {"path": src_parent_path}})
        except Exception as e:
            import traceback

            print(f"Error during notebook execution for {notebook_filename}:")
            traceback.print_exc()
            continue  # Continue with the next notebook if an error occurs

        # Export the executed notebook with outputs
        exporter = NotebookExporter()
        body, resources = exporter.from_notebook_node(notebook_content)

        # Write the executed notebook back to file
        with open(notebook_path, "w", encoding="utf-8") as f:
            write(notebook_content, f)

        print(f"Finished processing notebook: {notebook_path}")
