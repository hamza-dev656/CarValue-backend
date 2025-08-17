#!/usr/bin/env python3
"""
Test runner script for VinAudit backend integration tests
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests(test_category=None, verbose=False, coverage=False):
    """Run integration tests with optional filtering"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=70"
        ])
    
    # Add test category filter
    if test_category:
        if test_category == "api":
            cmd.append("tests/test_api_integration.py")
        elif test_category == "ml":
            cmd.append("tests/test_ml_integration.py")
        elif test_category == "cache":
            cmd.append("tests/test_cache_integration.py")
        elif test_category == "background":
            cmd.append("tests/test_background_training.py")
        elif test_category == "quick":
            cmd.extend(["-m", "not slow"])
        elif test_category == "slow":
            cmd.extend(["-m", "slow"])
    else:
        cmd.append("tests/")
    
    print("🧪 Running VinAudit Integration Tests")
    print("=" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Set environment variables for testing
    env = os.environ.copy()
    env.update({
        'TESTING': '1',
        'DATABASE_URL': 'sqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key',
        'CACHE_TYPE': 'SimpleCache',
        'CARVALUE_REUSE_TSV': '1',
        'CARVALUE_FORCE_REEXPORT': '0',
        'CARVALUE_ARTIFACT_DIR': './test_artifacts',
        'CARVALUE_DATA_DIR': './test_artifacts/data',
    })
    
    # Run tests
    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run VinAudit backend integration tests")
    
    parser.add_argument(
        '--category', '-c',
        choices=['api', 'ml', 'cache', 'background', 'quick', 'slow'],
        help='Run specific test category'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only check test setup without running tests'
    )
    
    args = parser.parse_args()
    
    if args.setup_only:
        print("🔧 Checking test setup...")
        
        # Check if pytest is available
        try:
            import pytest
            print(f"✅ pytest available (version {pytest.__version__})")
        except ImportError:
            print("❌ pytest not available - install with: pip install pytest")
            return 1
        
        # Check test directory structure
        test_dir = project_root / "tests"
        if test_dir.exists():
            test_files = list(test_dir.glob("test_*.py"))
            print(f"✅ Test directory found with {len(test_files)} test files")
        else:
            print("❌ Test directory not found")
            return 1
        
        # Check if app can be imported
        try:
            from app import create_app
            print("✅ App module can be imported")
        except Exception as e:
            print(f"❌ Cannot import app module: {e}")
            return 1
        
        print("✅ Test setup looks good!")
        return 0
    
    # Run tests
    return run_tests(args.category, args.verbose, args.coverage)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
