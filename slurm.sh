#!/bin/bash
#SBATCH --job-name=pure_react
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=5-00:00:00
#SBATCH --output=react_debug_%j.out
#SBATCH --error=react_debug_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu
#SBATCH --exclude=2115ga003

set -euo pipefail

module load cuda/12.4
source /cbica/projects/CXR/miniconda3/etc/profile.d/conda.sh
conda activate axolotl

REPO_PATH="/cbica/projects/CXR/codes/MIMIC-Clinical-Decision-Making-Framework"
DATA_ROOT_III="/cbica/projects/CXR/dropbox/CDM_III"
DATA_ROOT_IV="/cbica/projects/CXR/dropbox/CDM_IV"
PLAIN_REPO="/cbica/projects/CXR/codes/MIMIC-Plain"
LOG_DIR="${REPO_PATH}/outputs"

DISEASE=${1:-}
case "$DISEASE" in
    aortic_valve_disorders|mitral_valve_disorders|congestive_heart_failure|myocardial_infarction)
        DATA_PATH="$DATA_ROOT_III"
        REF_RANGES_JSON="${PLAIN_REPO}/itemid_ref_ranges_III.json"
        LAB_MAP_PKL="${PLAIN_REPO}/lab_test_mapping_III.pkl"
        ;;
    *)
        DATA_PATH="$DATA_ROOT_IV"
        REF_RANGES_JSON="${PLAIN_REPO}/itemid_ref_ranges_IV.json"
        LAB_MAP_PKL="${PLAIN_REPO}/lab_test_mapping_IV.pkl"
        ;;
esac

mkdir -p "$LOG_DIR"
cd "$REPO_PATH"

echo "Running disease: $DISEASE"
echo "Using data root: $DATA_PATH"
echo "Using lab mapping: $LAB_MAP_PKL"
echo "Using reference ranges: $REF_RANGES_JSON"
echo "Logging to: $LOG_DIR"

python run.py \
  --paths cbica \
  --pathology "$DISEASE" \
  --hadm-pkl "${DATA_PATH}/${DISEASE}.pkl" \
  --lab-map-pkl "$LAB_MAP_PKL" \
  --ref-ranges-json "$REF_RANGES_JSON" \
  --local-logging-dir "$LOG_DIR" \
  --hf-model-id meta-llama/Llama-3.3-70B-Instruct
