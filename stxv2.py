# Project Structure Generator.
"""
Usage:
- To run this script, place it in the root of the project's directory. 
- Use the following command to execute it:
    - python stxv2.py
- The script will generate a tree structure of the project and save it to a file named "project_stx.txt".

"""

import os

def generate_tree(directory, prefix="", level=0, max_level=4, exclude=None, output_file=None):
    if exclude is None:
        exclude = ["env", "venv", "__pycache__", ".git", "node_modules"]
    
    if level > max_level:
        return
    
    try:
        with os.scandir(directory) as it:  # Use scandir for efficiency
            entries = [entry for entry in it if entry.name not in exclude]
    except PermissionError:
        print(f"Permission denied: {directory}")
        return
    
    folders = sorted([entry.name for entry in entries if entry.is_dir()])
    files = sorted([entry.name for entry in entries if entry.is_file()])
    
    for folder in folders:
        line = f"{prefix}├── {folder}/\n"
        if output_file:
            output_file.write(line)
        print(line, end="")
        generate_tree(os.path.join(directory, folder), prefix + "│   ", level + 1, max_level, exclude, output_file)
    
    for i, file in enumerate(files):
        connector = "└──" if i == len(files) - 1 else "├──"
        line = f"{prefix}{connector} {file}\n"
        if output_file:
            output_file.write(line)
        print(line, end="")

# Enhanced Usage with error handling:
if __name__ == "__main__":
    root_dir = "."  # Use the current directory
    output_filename = "project_stx.txt"
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(os.path.basename(os.getcwd()) + "/\n")
            generate_tree(root_dir, output_file=f)
        print(f"\nTree structure saved to {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
