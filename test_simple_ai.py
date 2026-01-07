import os
import sys
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST: Simple AI-generated code")
print("=" * 60)

# Import what we need
from action_engine import ActionEngine, action_tool

# Create engine
engine = ActionEngine()

# SUPER SIMPLE request first
simple_request = "Create a text file called 'ai_test.txt' with the text 'Hello from AI' in it."

print(f"\\nTesting with request: {simple_request}")
print("This will use Gemini to generate code, then execute it...")

# Run the action_tool
result = action_tool(
    simple_request,
    engine,
    timeout=600  # Give it 10 minutes
)

print("\\n" + "=" * 60)
print("RESULT:")
print("=" * 60)
print(result)

# Check if file was created
if os.path.exists("ai_test.txt"):
    print("\\n[SUCCESS] File created!")
    with open("ai_test.txt", "r") as f:
        print(f"Contents: {f.read()}")
else:
    print("\\n[FAIL] File was not created")