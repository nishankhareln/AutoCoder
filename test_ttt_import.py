import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing TTT import...")
try:
    from ttt import Gemini
    print("[SUCCESS] TTT imported successfully!")
    
    # Test if we can create an instance
    print("Testing Gemini class creation...")
    g = Gemini(headless=False)
    print("[SUCCESS] Gemini instance created!")
    
except Exception as e:
    print(f"[FAILED] Error: {e}")
    import traceback
    traceback.print_exc()