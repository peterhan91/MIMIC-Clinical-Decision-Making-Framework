#!/bin/bash
#SBATCH --job-name=pure_react
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu.han@pennmedicine.upenn.edu

set -euo pipefail

# Apptainer container (from MIMIC-ReAct/slurm_1u.sh)
SIF="/cbica/home/hanti/ml_container.sif"
OVERLAY="/cbica/home/hanti/ml_overlay.img"
CONDA_ENV_PATH="/cbica/home/hanti/.conda/envs/ml310"

# Project paths
REPO="/cbica/projects/CXR/codes/MIMIC-Clinical-Decision-Making-Framework"
DATA_ROOT_III="/cbica/projects/CXR/dropbox/CDM_III"
DATA_ROOT_IV="/cbica/projects/CXR/dropbox/CDM_IV"
PLAIN_REPO="/cbica/projects/CXR/codes/MIMIC-Plain"
LOG_DIR="${REPO}/outputs"

# Caches and tmp (project-writable)
HF_HOME="/cbica/projects/CXR/.cache/huggingface"
XDG_CACHE_HOME="/cbica/projects/CXR/.cache"
NLTK_DATA="/cbica/projects/CXR/nltk_data"
JOB_TMP="/cbica/projects/CXR/scratch/SLURM_${SLURM_JOB_ID:-$$}"
TORCH_EXTENSIONS_DIR="${JOB_TMP}/torch_extensions"

mkdir -p "$LOG_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$JOB_TMP" "$TORCH_EXTENSIONS_DIR"

# Disease selection (arg1)
DISEASE="${1:-}"
if [[ -z "$DISEASE" ]]; then
  echo "Usage: sbatch slurm.sh <disease>" >&2
  exit 1
fi

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

HADM_PKL="${DATA_PATH}/${DISEASE}.pkl"

# Run inside Apptainer container
apptainer exec --nv \
  --bind /cbica:/cbica \
  ${OVERLAY:+--overlay "$OVERLAY"} \
  "$SIF" \
  bash -lc "
    set -euo pipefail
    # Activate conda inside container
    set +u
    if [ -f /opt/miniforge/etc/profile.d/conda.sh ]; then
      source /opt/miniforge/etc/profile.d/conda.sh
    elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      source /opt/conda/etc/profile.d/conda.sh
    else
      echo 'ERROR: conda.sh not found' >&2; exit 2
    fi
    conda activate '${CONDA_ENV_PATH}'
    set -u

    # Minimal env
    export HF_HOME='${HF_HOME}'
    export XDG_CACHE_HOME='${XDG_CACHE_HOME}'
    export HF_HUB_DISABLE_SYMLINKS_WARNING=1
    export NLTK_DATA='${NLTK_DATA}'
    export TMPDIR='${JOB_TMP}'
    export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}'

    cd '${REPO}'
    python run.py \
      --paths cbica \
      --pathology '${DISEASE}' \
      --hadm-pkl '${HADM_PKL}' \
      --lab-map-pkl '${LAB_MAP_PKL}' \
      --ref-ranges-json '${REF_RANGES_JSON}' \
      --local-logging-dir '${LOG_DIR}' \
      --hf-model-id meta-llama/Llama-3.3-70B-Instruct
  "
