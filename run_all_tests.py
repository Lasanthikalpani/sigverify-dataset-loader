"""
Run all SigVerify tests for supervisor demo
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a command and print result"""
    print("\n" + "=" * 70)
    print(f"[TEST] {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description}")
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"   {line}")
        else:
            print(f"[FAILED] {description}")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def main():
    print("=" * 70)
    print("SIGVERIFY - ALL TESTS")
    print("=" * 70)
    print("Supervisor Demo")
    print()
    
    tests = [
        ("python create_real_csv.py", "1. Create Sample CSV"),
        ("python src\\hybrid_dataset.py", "2. Generate Hybrid Dataset"),
        ("python src\\show_documents.py", "3. Show Original Documents"),
        ("python src\\load_real_data.py", "4. Load Real Data from CSV"),
        ("python src\\tamper_generator.py", "5. Test Tamper Detection"),
        ("python test_rq3_complete.py", "6. Complete RQ3 Test"),
        ("python test_gradcam.py", "7. Grad-CAM Test (RQ2)"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for desc, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status}  {desc}")
    
    print(f"\n  Total: {passed}/{total} passed")
    print("=" * 70)
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")


if __name__ == "__main__":
    main()