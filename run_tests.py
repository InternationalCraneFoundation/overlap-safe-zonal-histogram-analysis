"""
Test runner script for the zonal histogram analysis tool.

This script provides convenient commands for running different types of tests
and generating coverage reports.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run only unit tests
    python run_tests.py --integration # Run only integration tests
    python run_tests.py --coverage   # Run tests with coverage report
"""

import subprocess
import sys
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure the development environment is activated")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for zonal histogram tool")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Base pytest command - use sys.executable to ensure same Python
    pytest_cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        pytest_cmd.append("-v")

    # Determine which tests to run
    if args.unit:
        pytest_cmd.extend(["-m", "unit"])
        description = "Unit Tests"
    elif args.integration:
        pytest_cmd.extend(["-m", "integration"])
        description = "Integration Tests"
    else:
        description = "All Tests"

    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
        description += " with Coverage"

    # Run the tests
    success = run_command(pytest_cmd, description)

    if args.coverage and success:
        print("\n📊 Coverage report generated in htmlcov/index.html")

    # Run linting if all tests pass
    if success and not (args.unit or args.integration):
        print("\n🔍 Running code quality checks...")

        # Run ruff for linting
        if run_command([sys.executable, "-m", "ruff", "check", "."], "Ruff Linting"):
            print("✅ No linting issues found")

        # Run black for formatting check
        if run_command([sys.executable, "-m", "black", "--check", "."], "Black Formatting Check"):
            print("✅ Code formatting is correct")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
