import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from action_engine import ActionEngine

engine = ActionEngine()

# Simple test
code = '''print("Test execution...")
with open("test_exec.txt", "w") as f:
    f.write("Working!")
print("ACTIONENGINE: OK - File created")'''

result, _ = engine.run_code_block(code, timeout=10)
print(f"Return: {result.returncode}")
print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")

if os.path.exists("test_exec.txt"):
    print("✓ File created!")