import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from action_engine import format_and_autofix_code

# Test with bad code (like what Gemini generates)
bad_code = '''import pandas as pd
from datetime import datetime, timedelta
import random

try:
 products = ['Widget A', 'Gadget B', 'Tool C']
 end_date = datetime.now()
 start_date = end_date - timedelta(days=30)
'''

print("Testing formatter...")
formatted, report = format_and_autofix_code(bad_code)

print("\\n=== ORIGINAL ===")
print(repr(bad_code))
print("\\n=== FORMATTED ===")
print(repr(formatted))
print("\\n=== REPORT ===")
print(f"AST OK before: {report.get('ast_ok_before')}")
print(f"AST OK after: {report.get('final_ast_ok')}")
if report.get('errors'):
    print(f"Errors: {report.get('errors')}")
if report.get('warnings'):
    print(f"Warnings: {report.get('warnings')}")

# Try to parse it
import ast
try:
    ast.parse(formatted)
    print("\\n✓ Formatted code parses successfully!")
except SyntaxError as e:
    print(f"\\n✗ Formatted code has syntax error: {e}")