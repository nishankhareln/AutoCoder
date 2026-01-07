from action_engine import ActionEngine

print("=" * 60)
print("TEST 1: Basic ActionEngine test (no AI)")
print("=" * 60)

# Create engine
engine = ActionEngine()

# Simple test code
test_code = '''
print("=== Starting basic test ===")
print("Testing Python imports...")

try:
    import requests
    print(f"[OK] Requests imported: {requests.__version__}")
except Exception as e:
    print(f"[FAIL] Requests failed: {e}")

try:
    import pandas as pd
    print(f"[OK] Pandas imported: {pd.__version__}")
except Exception as e:
    print(f"[FAIL] Pandas failed: {e}")

try:
    import numpy as np
    print(f"[OK] Numpy imported: {np.__version__}")
except Exception as e:
    print(f"[FAIL] Numpy failed: {e}")

print("=== Creating test file ===")
with open("test_output.txt", "w") as f:
    f.write("Hello from ActionEngine test\\nThis is working!")
    print("[OK] Created test_output.txt")

print("ACTIONENGINE: OK - Basic test completed successfully")
print("=== Test finished ===")
'''

print("\\nRunning test code...")
result, _ = engine.run_code_block(test_code, timeout=60)
print("\\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Return code: {result.returncode}")
print("\\n--- STDOUT ---")
print(result.stdout)
if result.stderr:
    print("\\n--- STDERR ---")
    print(result.stderr)

# Check for the file
import os
if os.path.exists("test_output.txt"):
    print("\\n[OK] File was created successfully!")
    with open("test_output.txt", "r") as f:
        print(f"Contents: {f.read()}")
else:
    print("\\n[FAIL] File was NOT created")