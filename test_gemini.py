import os
import sys
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST: ActionEngine + Gemini Connection")
print("=" * 60)

# Test 1: Import both modules
try:
    from action_engine import ActionEngine
    from ttt import Gemini
    print("[OK] Both modules imported")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create ActionEngine instance
try:
    engine = ActionEngine()
    print("[OK] ActionEngine created")
except Exception as e:
    print(f"[FAIL] ActionEngine creation: {e}")

# Test 3: Simple test of Gemini
async def test_gemini():
    try:
        print("\\nStarting Gemini test (headless=False so you can see it)...")
        async with Gemini(headless=False) as g:
            print("[OK] Gemini session started")
            
            # Simple request
            print("Sending request to Gemini...")
            result = await g.send(
                user_prompt="Say 'Connection test successful' in one sentence.",
                system_prompt="Keep response very short. Wrap your response in <response> tags.",
                return_blocks=["response"],
                stream=False
            )
            
            if result and "blocks" in result:
                response = result["blocks"].get("response", ["No response"])[0]
                print(f"[OK] Gemini responded: {response}")
                return True
            else:
                print("[FAIL] No response blocks")
                return False
    except Exception as e:
        print(f"[FAIL] Gemini test error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the async test
try:
    success = asyncio.run(test_gemini())
    if success:
        print("\\n[SUCCESS] Gemini connection test PASSED ✓")
    else:
        print("\\n[FAIL] Gemini connection test FAILED ✗")
except Exception as e:
    print(f"\\n[ERROR] Async error: {e}")

print("\\n" + "=" * 60)
print("Test complete!")
print("=" * 60)