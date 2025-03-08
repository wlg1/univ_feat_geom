#!/bin/bash
#
# run_pythia.sh
#
# A professional script to run the Pythia model comparisons.
#
# Usage:
#   ./run_pythia.sh [options]
#
# Options:
#   -b, --batch_size           Batch size (default: 400)
#   -m, --max_length           Maximum sequence length (default: 400)
#   -r, --num_rand_runs        Number of random runs (default: 1)
#   -1, --oneToOne_bool        Set one-to-one flag (default: false)
#   -a, --model_A_endLayer     End layer for Model A (default: 6)
#   -B, --model_B_endLayer     End layer for Model B (default: 12)
#   -s, --layer_step_size      Layer step size (default: 2)
#   -h, --help                 Display this help message
#
# Example:
#   ./run_pythia.sh --batch_size 400 --max_length 400 --num_rand_runs 1 --oneToOne_bool --model_A_endLayer 6 --model_B_endLayer 12 --layer_step_size 2
#

set -e  # Exit immediately if a command exits with a non-zero status

# Default parameter values
BATCH_SIZE=400
MAX_LENGTH=400
NUM_RAND_RUNS=1
ONE_TO_ONE_BOOL=false
MODEL_A_ENDLAYER=6
MODEL_B_ENDLAYER=12
LAYER_STEP_SIZE=2

function usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -b, --batch_size           Batch size (default: ${BATCH_SIZE})"
    echo "  -m, --max_length           Maximum sequence length (default: ${MAX_LENGTH})"
    echo "  -r, --num_rand_runs        Number of random runs (default: ${NUM_RAND_RUNS})"
    echo "  -1, --oneToOne_bool        Set one-to-one flag (default: false)"
    echo "  -a, --model_A_endLayer     End layer for Model A (default: ${MODEL_A_ENDLAYER})"
    echo "  -B, --model_B_endLayer     End layer for Model B (default: ${MODEL_B_ENDLAYER})"
    echo "  -s, --layer_step_size      Layer step size (default: ${LAYER_STEP_SIZE})"
    echo "  -h, --help                 Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift
            ;;
        -m|--max_length)
            MAX_LENGTH="$2"
            shift
            ;;
        -r|--num_rand_runs)
            NUM_RAND_RUNS="$2"
            shift
            ;;
        -1|--oneToOne_bool)
            ONE_TO_ONE_BOOL=true
            ;;
        -a|--model_A_endLayer)
            MODEL_A_ENDLAYER="$2"
            shift
            ;;
        -B|--model_B_endLayer)
            MODEL_B_ENDLAYER="$2"
            shift
            ;;
        -s|--layer_step_size)
            LAYER_STEP_SIZE="$2"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
    shift
done

echo "Starting Pythia model comparison run with the following parameters:"
echo "  Batch size:            ${BATCH_SIZE}"
echo "  Maximum length:        ${MAX_LENGTH}"
echo "  Number of random runs: ${NUM_RAND_RUNS}"
echo "  One-to-One flag:       ${ONE_TO_ONE_BOOL}"
echo "  Model A end layer:     ${MODEL_A_ENDLAYER}"
echo "  Model B end layer:     ${MODEL_B_ENDLAYER}"
echo "  Layer step size:       ${LAYER_STEP_SIZE}"

# Optionally, activate your virtual environment here if needed:
# source /path/to/venv/bin/activate

# Construct the one-to-one flag argument (only include if true)
ONE_TO_ONE_ARG=""
if [ "${ONE_TO_ONE_BOOL}" = true ]; then
    ONE_TO_ONE_ARG="--oneToOne_bool"
fi

# Run the Python script with the provided parameters
python run.py --batch_size "${BATCH_SIZE}" \
              --max_length "${MAX_LENGTH}" \
              --num_rand_runs "${NUM_RAND_RUNS}" \
              ${ONE_TO_ONE_ARG} \
              --model_A_endLayer "${MODEL_A_ENDLAYER}" \
              --model_B_endLayer "${MODEL_B_ENDLAYER}" \
              --layer_step_size "${LAYER_STEP_SIZE}"

echo "Pythia model comparison run completed successfully."
