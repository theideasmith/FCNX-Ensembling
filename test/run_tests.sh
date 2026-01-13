#!/bin/bash
# ============================================================================
# run_tests.sh — Run all tests in the FCNX-Ensembling project
# ============================================================================
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh -v           # Run with verbose output
#   ./run_tests.sh -k <pattern> # Run tests matching pattern
#   ./run_tests.sh --coverage   # Run with coverage report
#
# ============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Change to project root
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Running FCNX-Ensembling Test Suite${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Parse arguments
PYTEST_ARGS="-v"
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            PYTEST_ARGS="-vv"
            shift
            ;;
        -k)
            PYTEST_ARGS="$PYTEST_ARGS -k $2"
            shift 2
            ;;
        -x|--exitfirst)
            PYTEST_ARGS="$PYTEST_ARGS -x"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: ./run_tests.sh [options]"
            echo "Options:"
            echo "  --coverage     Run with coverage report"
            echo "  -v, --verbose  Verbose output"
            echo "  -k PATTERN     Run tests matching pattern"
            echo "  -x             Exit on first failure"
            exit 1
            ;;
    esac
done

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest"
    exit 1
fi

# Run tests with optional coverage
if [ "$COVERAGE" = true ]; then
    if ! command -v pytest &> /dev/null; then
        echo -e "${RED}Error: pytest-cov not found${NC}"
        echo "Install with: pip install pytest-cov"
        exit 1
    fi
    
    echo -e "${YELLOW}Running tests with coverage...${NC}"
    echo ""
    pytest $PYTEST_ARGS --cov=lib --cov-report=html --cov-report=term-missing test/
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo -e "${YELLOW}Running tests...${NC}"
    echo ""
    pytest $PYTEST_ARGS test/
fi

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${YELLOW}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Tests failed (exit code: $EXIT_CODE)${NC}"
fi

echo -e "${YELLOW}========================================${NC}"

exit $EXIT_CODE
