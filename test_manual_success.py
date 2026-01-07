import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from action_engine import ActionEngine

print("=" * 60)
print("MANUAL TEST: Create sales file with CORRECT indentation")
print("=" * 60)

engine = ActionEngine()

# Manually write CORRECTLY indented code
correct_code = '''import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    products = ["Product A", "Product B", "Product C"]
    dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(30)]
    
    data = []
    for date in dates:
        for product in products:
            data.append({
                "Date": date,
                "Product": product,
                "Sales": np.random.randint(10, 100),
                "Revenue": round(np.random.uniform(100.0, 500.0), 2)
            })
    
    df = pd.DataFrame(data)
    df.to_csv("sales_report.csv", index=False)
    print("ACTIONENGINE: OK - sales_report.csv created")
except Exception as e:
    print(f"ACTIONENGINE: ERROR - {e}")
'''

print("\\nExecuting code...")
result, _ = engine.run_code_block(correct_code, timeout=30)

print(f"\\nReturn code: {result.returncode}")
print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")

# Check if file was created
if os.path.exists("sales_report.csv"):
    print("\\n✅ SUCCESS! File created!")
    # Show a preview
    try:
        import pandas as pd
        df = pd.read_csv("sales_report.csv")
        print(f"\\nFirst 3 rows:")
        print(df.head(3))
        print(f"\\nFile saved at: {os.path.abspath('sales_report.csv')}")
    except:
        print("Could not read CSV, but file exists")
else:
    print("\\n❌ FAILED: File not created")