#!/bin/bash

set -xe

BASE_DIR=$(dirname $(readlink -f "$0"))

# Function to display usage
usage() {
  echo "Usage: $0 -f <mlir_file> -m <module_name>"
  echo ""
  echo "Options:"
  echo "  -f, --file <mlir_file>      Specify the input MLIR file (required)."
  echo "  -m, --module <module_name>  Specify the top module name in the MLIR file (required)."
  echo "  -h, --help                  Display this help message."
  exit 1
}

# Initialize variables
MLIR_FILE=""
MODULE_NAME=""

# Parse command-line options using getopt
# Use getopt command (external utility) for long options support
TEMP=$(getopt -o 'hf:m:' --long 'help,file:,module:' -n "$0" -- "$@")
if [ $? != 0 ]; then
  echo "Error parsing options. Terminating..." >&2
  usage
fi

eval set -- "$TEMP"
unset TEMP

while true; do
  case "$1" in
    '-f' | '--file')
      MLIR_FILE="$2"
      shift 2
      continue
      ;;
    '-m' | '--module')
      MODULE_NAME="$2"
      shift 2
      continue
      ;;
    '-h' | '--help')
      usage
      ;;
    '--')
      shift
      break
      ;;
    *)
      echo 'Internal error!' >&2
      exit 1
      ;;
  esac
done

# Check if mandatory options were provided
if [ -z "$MLIR_FILE" ]; then
  echo "Error: MLIR file not specified." >&2
  usage
fi

if [ -z "$MODULE_NAME" ]; then
  echo "Error: Module name not specified." >&2
  usage
fi

# Define WORK_DIR after arguments are parsed and validated
WORK_DIR="$(pwd)/difftest_simdir" # Use $(pwd) for better practice

# Check if the specified MLIR file exists
if [ ! -f "$MLIR_FILE" ]; then
    echo "Error: File '$MLIR_FILE' does not exist." >&2
    exit 1
fi

# Check if the specified module name exists in the MLIR file
# Use quotes around variables for robustness
if ! grep -q "hw.module @${MODULE_NAME}(" "$MLIR_FILE"; then
    echo "Error: Module name '$MODULE_NAME' does not exist in the MLIR file '$MLIR_FILE'." >&2
    exit 1
fi

rm -rf $WORK_DIR
mkdir $WORK_DIR

# Generate golden module
circt-opt "$MLIR_FILE" --convert-xlnx-to-hw --lower-seq-to-sv | firtool -format mlir | sed "s/\b${MODULE_NAME}\b/GoldenTop/g" > $WORK_DIR/golden.sv

# Generate under test module
circt-opt "$MLIR_FILE" --convert-seq-to-xlnx --convert-xlnx-to-hw --split-input-file --verify-diagnostics | firtool -format=mlir | sed "s/\b${MODULE_NAME}\b/CheckTop/g" > $WORK_DIR/under_test.sv

# Run difference test
pushd $WORK_DIR

# Assume the test platform (tb.sv) instantiates GoldenTop and CheckTop
iverilog -o diff_test -g2012 -s tb $BASE_DIR/tb.sv $BASE_DIR/prim_cell_lib.v golden.sv under_test.sv
vvp diff_test

popd
