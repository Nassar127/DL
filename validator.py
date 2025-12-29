import os
import json
import numpy as np

def run_qa_audit(prediction_dir):
    print("\n" + "="*50)
    print("      EXPERT 5: OUTPUT VALIDATION AUDIT      ")
    print("="*50)
    
    if not os.path.exists(prediction_dir):
        print(f"❌ ERROR: Prediction folder '{prediction_dir}' not found!")
        return

    json_files = [f for f in os.listdir(prediction_dir) if f.endswith('.json')]
    print(f"🔍 Found {len(json_files)} prediction files. Starting scan...")

    issues = 0
    for filename in json_files:
        with open(os.path.join(prediction_dir, filename), 'r') as f:
            try:
                data = json.load(f)
                
                # 1. فحص إحداثيات الصناديق (Bounding Box Bounds)
                if 'boxes' in data:
                    for box in data['boxes']:
                        if any(c < 0 or c > 1 for c in box):
                            print(f"⚠️  COORD ERROR in {filename}: Box {box} is out of normalized range (0-1)!")
                            issues += 1
                
                # 2. فحص مسار القناع (Mask Path Integrity)
                if 'mask_path' in data:
                    mask_full_path = os.path.join(prediction_dir, data['mask_path'])
                    if not os.path.exists(mask_full_path):
                        print(f"⚠️  FILE ERROR in {filename}: Mask image '{data['mask_path']}' is missing!")
                        issues += 1
                        
            except Exception as e:
                print(f"❌ CRITICAL: Could not read {filename}. Error: {e}")
                issues += 1

    print("="*50)
    if issues == 0:
        print("✅ AUDIT PASSED: All outputs are safe for submission.")
    else:
        print(f"❌ AUDIT FAILED: Found {issues} issues. Do not submit yet!")
    print("="*50 + "\n")

if __name__ == "__main__":
    # هذا المجلد الذي سيخرج فيه التدريب نتائجه
    run_qa_audit("./predictions_new_new")