"""
TRATA - Dependency Checker
----------------------------------
Run this before the pipeline to verify all required packages are installed.
"""

import sys


def check_dependency(name, import_name=None):
    """Try to import a package and report status."""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"  ✓ {name:20} installed")
        return True
    except ImportError:
        print(f"  ✗ {name:20} NOT FOUND — run: pip install {name}")
        return False


def main():
    print("\nAdityaNetra — Dependency Checker\n")
    print("Checking required packages...\n")

    deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
    ]

    all_ok = all(check_dependency(name, imp) for name, imp in deps)

    print()
    if all_ok:
        print("✓ All dependencies installed! You're ready to run the pipeline.")
        print("\n  python3 run_pipeline.py")
        return 0
    else:
        print("✗ Some dependencies are missing. Install them with:")
        print("\n  python3 -m pip install torch numpy pandas scipy scikit-learn joblib")
        return 1


if __name__ == "__main__":
    sys.exit(main())
