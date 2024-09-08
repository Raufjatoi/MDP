import os
from sys import builtin_module_names
from pkgutil import iter_modules

standard_libs = [m.name.replace('_', '') for m in iter_modules()] + list(builtin_module_names)

# Make a list of all the Python files present
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
                # Handle `import lib1 as alias1, lib2 as alias2` or without aliases
                libs = line.split('import ')[1].split(',')
                for lib in libs:
                    lib = lib.split('as')[0].split('.')[0].strip()  # Handle 'as' and ignore aliases
                    imports.add(lib)
                    
            elif line.startswith('from '):
                # Handle `from lib import something`
                lib = line.split(' ')[1].split('.')[0].strip()
                imports.add(lib)    

#Clear out non-std libs
non_std = []
for lib in imports:
    if lib not in standard_libs:
        non_std.append(lib)

# Write the extracted libraries to requirements.txt
with open('requirements.txt', 'w') as file:
    for lib in non_std:
        file.write(lib + "\n")
