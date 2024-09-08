# This assumes you are running the program in the root 
# directory of the project
import os

# Make a list of all the python files present
files = []
# Walk through the directory tree
for dirpath, _, filenames in os.walk('.'):
    for filename in filenames:
        # Check if the file is a Python file
        if filename.endswith('.py'):
            # Get the full path to the file and append to the list
            full_path = os.path.join(dirpath, filename)
            files.append(full_path)
            
"""Extract libraries from lines containing 'import'."""
imports = set()
for file_path in files:
  with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        
        if line.startswith('import '):
           # Handle `import lib1, lib2`
           libs = line.split('import ')[1].split(',')
           for lib in libs:
                  imports.add(lib.split('.')[0].strip())
                  
        elif line.startswith('from '):
            # Handle `from lib import something`
            lib = line.split(' ')[1].split('.')[0].strip()
            imports.add(lib)    
            
with open('requirements.txt', 'w') as file:
    for lib in imports:
        file.write(lib + "\n")                   
