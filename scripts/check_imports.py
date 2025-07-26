#!/usr/bin/env python3
"""
Import Style Checker for PiCor

This script checks import statements for compliance with PEP 8 and common Python conventions.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
PURPLE = '\033[0;35m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color

def print_status(msg: str) -> None:
    """Print status message."""
    print(f"{BLUE}[INFO]{NC} {msg}")

def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}[SUCCESS]{NC} {msg}")

def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}[WARNING]{NC} {msg}")

def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{RED}[ERROR]{NC} {msg}")

def print_header(msg: str) -> None:
    """Print header message."""
    print(f"{BOLD}{CYAN}{msg}{NC}")

class ImportChecker:
    """Check import statements for style compliance."""
    
    def __init__(self):
        self.issues = []
        self.total_files = 0
        self.files_with_issues = 0
        
    def check_import_order(self, imports: List[Union[ast.Import, ast.ImportFrom]]) -> List[str]:
        """Check if imports are in the correct order."""
        issues = []
        
        # Expected order: stdlib, third-party, local
        current_section = 0  # 0: stdlib, 1: third-party, 2: local
        
        for i, imp in enumerate(imports):
            if isinstance(imp, ast.Import):
                module_name = imp.names[0].name
            else:
                module_name = imp.module or ""
            
            # Determine section
            if self._is_stdlib_module(module_name):
                section = 0
            elif self._is_third_party_module(module_name):
                section = 1
            else:
                section = 2
            
            # Check order
            if section < current_section:
                issues.append(f"Import '{module_name}' should come before other imports (line {imp.lineno})")
            elif section > current_section:
                current_section = section
                
        return issues
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if module is from Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'time', 'random', 'datetime', 'pickle', 'json', 'typing',
            'argparse', 'pathlib', 'collections', 'itertools', 'functools',
            'math', 'statistics', 'copy', 'weakref', 'abc', 'enum'
        }
        return module_name.split('.')[0] in stdlib_modules
    
    def _is_third_party_module(self, module_name: str) -> bool:
        """Check if module is a third-party library."""
        third_party_modules = {
            'torch', 'numpy', 'gym', 'wandb', 'tensordict', 'torchrl',
            'yaml', 'metaworld', 'garage', 'dotenv'
        }
        return module_name.split('.')[0] in third_party_modules
    
    def check_import_spacing(self, imports: List[Union[ast.Import, ast.ImportFrom]]) -> List[str]:
        """Check spacing between import groups."""
        issues = []
        
        for i in range(1, len(imports)):
            prev_imp = imports[i-1]
            curr_imp = imports[i]
            
            # Check if there should be a blank line between different sections
            prev_module = self._get_module_name(prev_imp)
            curr_module = self._get_module_name(curr_imp)
            
            prev_section = self._get_section(prev_module)
            curr_section = self._get_section(curr_module)
            
            if curr_section > prev_section and curr_imp.lineno - prev_imp.lineno <= 1:
                issues.append(f"Missing blank line between import sections (line {curr_imp.lineno})")
                
        return issues
    
    def _get_module_name(self, imp: Union[ast.Import, ast.ImportFrom]) -> str:
        """Get module name from import statement."""
        if isinstance(imp, ast.Import):
            return imp.names[0].name
        return imp.module or ""
    
    def _get_section(self, module_name: str) -> int:
        """Get section number for module."""
        if self._is_stdlib_module(module_name):
            return 0
        elif self._is_third_party_module(module_name):
            return 1
        else:
            return 2
    
    def check_import_style(self, imports: List[Union[ast.Import, ast.ImportFrom]], file_path: Path) -> List[str]:
        """Check import style issues."""
        issues = []
        
        for imp in imports:
            if isinstance(imp, ast.Import):
                # Check for wildcard imports
                for alias in imp.names:
                    if alias.name == '*':
                        issues.append(f"Wildcard import detected (line {imp.lineno})")
                        
                # Check for multiple imports on one line
                if len(imp.names) > 1:
                    issues.append(f"Multiple imports on one line (line {imp.lineno})")
                    
            elif isinstance(imp, ast.ImportFrom):
                # Check for relative imports (allow in __init__.py files)
                if imp.level > 0 and file_path.name != '__init__.py':
                    issues.append(f"Relative import detected (line {imp.lineno})")
                    
                # Check for wildcard imports
                for alias in imp.names:
                    if alias.name == '*':
                        issues.append(f"Wildcard import detected (line {imp.lineno})")
        
        return issues
    
    def check_file(self, file_path: Path) -> List[str]:
        """Check import statements in a single file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Find all import statements
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
            
            # Sort imports by line number
            imports.sort(key=lambda x: x.lineno)
            
            # Check various import issues
            issues.extend(self.check_import_order(imports))
            issues.extend(self.check_import_spacing(imports))
            issues.extend(self.check_import_style(imports, file_path))
            
        except Exception as e:
            issues.append(f"Error parsing file: {e}")
            
        return issues
    
    def check_directory(self, directory: Path) -> None:
        """Check all Python files in a directory."""
        print_header(f"Checking imports in {directory}")
        
        python_files = list(directory.rglob("*.py"))
        self.total_files = len(python_files)
        
        for file_path in python_files:
            issues = self.check_file(file_path)
            
            if issues:
                self.files_with_issues += 1
                print_warning(f"{file_path}")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print_success(f"{file_path}")
    
    def print_summary(self) -> None:
        """Print summary of import check results."""
        print_header("\n📊 Import Check Summary")
        
        if self.files_with_issues == 0:
            print_success(f"All {self.total_files} files have clean imports!")
        else:
            print_warning(f"{self.files_with_issues}/{self.total_files} files have import issues")
            
        print_status("💡 Import Style Guidelines:")
        print("  1. Group imports: stdlib → third-party → local")
        print("  2. Add blank lines between import groups")
        print("  3. Avoid wildcard imports (*)")
        print("  4. Avoid multiple imports on one line")
        print("  5. Use absolute imports over relative imports")

def main():
    """Main function to run import checks."""
    checker = ImportChecker()
    
    # Check source directory
    source_dir = Path("source")
    if source_dir.exists():
        checker.check_directory(source_dir)
    else:
        print_error("Source directory not found")
        sys.exit(1)
    
    checker.print_summary()
    
    # Exit with error code if issues found
    if checker.files_with_issues > 0:
        sys.exit(1)

if __name__ == "__main__":
    main() 