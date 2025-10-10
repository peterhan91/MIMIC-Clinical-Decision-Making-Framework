import argparse
import json
import os
from os.path import join
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import langchain
import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM
from agents.agent import build_agent_executor_ZeroShot

HF_ID_TO_MODEL_CONFIG = {
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama3Instruct70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama3.1Instruct70B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3Instruct70B",
    "aaditya/OpenBioLLM-Llama3-70B": "OpenBioLLM70B",
    "axiong/PMC_LLaMA_13B": "PMCLlama13B",
}

CLI_ADAPTATION_WARNINGS = []


def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


def _quote_for_hydra(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_override(key: str, value: str, quote: bool = False) -> str:
    if value is None:
        return f"{key}=null"
    return f"{key}={_quote_for_hydra(value) if quote else value}"


def _parse_ref_range_entry(entry):
    if isinstance(entry, dict):
        lower = (
            entry.get("lower")
            or entry.get("low")
            or entry.get("ref_range_lower")
            or entry.get("min")
        )
        upper = (
            entry.get("upper")
            or entry.get("high")
            or entry.get("ref_range_upper")
            or entry.get("max")
        )
        if lower is None:
            for key, val in entry.items():
                if "low" in key.lower() or "min" in key.lower():
                    lower = val
                    break
        if upper is None:
            for key, val in entry.items():
                if "high" in key.lower() or "max" in key.lower():
                    upper = val
                    break
        return lower, upper
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[0], entry[1]
    return None, None


def _apply_reference_ranges(hadm_info_clean, ref_ranges_json_path: str):
    if not ref_ranges_json_path:
        return
    if not os.path.exists(ref_ranges_json_path):
        CLI_ADAPTATION_WARNINGS.append(
            f"Reference range file not found: {ref_ranges_json_path}"
        )
        return
    with open(ref_ranges_json_path, "r") as handle:
        ref_data = json.load(handle)
    parsed_ranges = {}
    for raw_key, entry in ref_data.items():
        try:
            itemid_int = int(raw_key)
        except (ValueError, TypeError):
            itemid_int = raw_key
        lower, upper = _parse_ref_range_entry(entry)
        if lower is None or upper is None:
            continue
        parsed_ranges[itemid_int] = (lower, upper)
    if not parsed_ranges:
        CLI_ADAPTATION_WARNINGS.append(
            f"No usable reference ranges parsed from {ref_ranges_json_path}"
        )
        return
    for hadm_entry in hadm_info_clean.values():
        lower_dict = hadm_entry.setdefault("Reference Range Lower", {})
        upper_dict = hadm_entry.setdefault("Reference Range Upper", {})
        for itemid, (lower, upper) in parsed_ranges.items():
            if itemid not in lower_dict:
                lower_dict[itemid] = lower
            if itemid not in upper_dict:
                upper_dict[itemid] = upper
            itemid_str = str(itemid)
            if itemid_str not in lower_dict:
                lower_dict[itemid_str] = lower
            if itemid_str not in upper_dict:
                upper_dict[itemid_str] = upper


def _load_patient_data(args: DictConfig):
    hadm_pickle = getattr(args, "hadm_pickle_path", None)
    base_mimic = getattr(args, "base_mimic", "")
    if hadm_pickle:
        hadm_path = hadm_pickle
        if not os.path.isabs(hadm_path) and base_mimic:
            hadm_path = join(base_mimic, hadm_path)
        with open(hadm_path, "rb") as handle:
            hadm_info_clean = pickle.load(handle)
    else:
        hadm_info_clean = load_hadm_from_file(
            f"{args.pathology}_hadm_info_first_diag", base_mimic=base_mimic
        )
    ref_ranges_json_path = getattr(args, "ref_ranges_json_path", "")
    _apply_reference_ranges(hadm_info_clean, ref_ranges_json_path)
    return hadm_info_clean


def _adapt_slurm_cli_args():
    global CLI_ADAPTATION_WARNINGS
    if len(sys.argv) <= 1:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paths")
    parser.add_argument("--pathology")
    parser.add_argument("--hadm-pkl")
    parser.add_argument("--lab-map-pkl")
    parser.add_argument("--ref-ranges-json")
    parser.add_argument("--hf-model-id")
    parser.add_argument("--agent-type")
    parser.add_argument("--rewoo-planner-reflexion", action="store_true")
    parser.add_argument("--rewoo-num-plans", type=int)
    parser.add_argument("--include-ref-range", action="store_true")
    parser.add_argument("--bin-lab-results", action="store_true")
    parser.add_argument("--use-calculator", action="store_true")
    parser.add_argument("--calculator-include-units", action="store_true")
    parser.add_argument("--local-logging-dir")
    parser.add_argument("--base-model-cache")
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    recognized = any(
        [
            parsed.paths,
            parsed.pathology,
            parsed.hadm_pkl,
            parsed.lab_map_pkl,
            parsed.ref_ranges_json,
            parsed.hf_model_id,
            parsed.agent_type,
            parsed.rewoo_planner_reflexion,
            parsed.rewoo_num_plans is not None,
            parsed.include_ref_range,
            parsed.bin_lab_results,
            parsed.use_calculator,
            parsed.calculator_include_units,
            parsed.local_logging_dir,
            parsed.base_model_cache,
        ]
    )
    if not recognized:
        return

    overrides = []

    if parsed.paths:
        overrides.append(_format_override("paths", parsed.paths))
    elif parsed.hadm_pkl and "/cbica/" in parsed.hadm_pkl:
        overrides.append(_format_override("paths", "cbica"))

    if parsed.hadm_pkl:
        overrides.append(
            _format_override("hadm_pickle_path", parsed.hadm_pkl, quote=True)
        )
        inferred_pathology = parsed.pathology or Path(parsed.hadm_pkl).stem
        overrides.append(
            _format_override("pathology", inferred_pathology.replace("-", "_"))
        )
    elif parsed.pathology:
        overrides.append(
            _format_override("pathology", parsed.pathology.replace("-", "_"))
        )

    if parsed.lab_map_pkl:
        overrides.append(
            _format_override("lab_test_mapping_path", parsed.lab_map_pkl, quote=True)
        )

    if parsed.ref_ranges_json:
        overrides.append(
            _format_override("ref_ranges_json_path", parsed.ref_ranges_json, quote=True)
        )

    if parsed.hf_model_id:
        mapped_model = HF_ID_TO_MODEL_CONFIG.get(parsed.hf_model_id)
        if mapped_model:
            overrides.append(_format_override("model", mapped_model))
        else:
            overrides.append(
                _format_override("model_name", parsed.hf_model_id, quote=True)
            )
            CLI_ADAPTATION_WARNINGS.append(
                f"No pre-defined model config for {parsed.hf_model_id}; "
                "using direct model_name override."
            )

    if parsed.agent_type:
        if parsed.agent_type.lower() == "zeroshot":
            overrides.append(_format_override("agent", "ZeroShot"))
        elif parsed.agent_type.lower() == "rewoo":
            overrides.append(_format_override("agent", "ZeroShot"))
            CLI_ADAPTATION_WARNINGS.append(
                "Agent type 'rewoo' requested but not implemented; falling back to ZeroShot."
            )
        else:
            overrides.append(_format_override("agent", parsed.agent_type))

    if parsed.include_ref_range:
        overrides.append("include_ref_range=true")
    if parsed.bin_lab_results:
        overrides.append("bin_lab_results=true")
    if parsed.local_logging_dir:
        overrides.append(
            _format_override("local_logging_dir", parsed.local_logging_dir, quote=True)
        )
    if parsed.base_model_cache:
        overrides.append(
            _format_override("base_models", parsed.base_model_cache, quote=True)
        )

    if parsed.rewoo_planner_reflexion:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --rewoo-planner-reflexion (feature not available in this repo)."
        )
    if parsed.rewoo_num_plans is not None:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --rewoo-num-plans (feature not available in this repo)."
        )
    if parsed.use_calculator:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --use-calculator (feature not available in this repo)."
        )
    if parsed.calculator_include_units:
        CLI_ADAPTATION_WARNINGS.append(
            "Ignoring --calculator-include-units (feature not available in this repo)."
        )

    sys.argv = [sys.argv[0]] + overrides + remaining


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    if not args.self_consistency:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load patient data
    hadm_info_clean = _load_patient_data(args)

    tags = {
        "system_tag_start": args.system_tag_start,
        "user_tag_start": args.user_tag_start,
        "ai_tag_start": args.ai_tag_start,
        "system_tag_end": args.system_tag_end,
        "user_tag_end": args.user_tag_end,
        "ai_tag_end": args.ai_tag_end,
    }

    # Load desired model
    llm = CustomLLM(
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        tags=tags,
        max_context_length=args.max_context_length,
        exllama=args.exllama,
        seed=args.seed,
        self_consistency=args.self_consistency,
    )
    llm.load_model(args.base_models)

    date_time = datetime.fromtimestamp(time.time())
    str_date = date_time.strftime("%d-%m-%Y_%H:%M:%S")
    args.model_name = args.model_name.replace("/", "_")
    run_name = f"{args.pathology}_{args.agent}_{args.model_name}_{str_date}"
    if args.fewshot:
        run_name += "_FEWSHOT"
    if args.include_ref_range:
        if args.bin_lab_results:
            raise ValueError(
                "Binning and printing reference ranges concurrently is not supported."
            )
        run_name += "_REFRANGE"
    if args.bin_lab_results:
        run_name += "_BIN"
    if args.include_tool_use_examples:
        run_name += "_TOOLEXAMPLES"
    if args.provide_diagnostic_criteria:
        run_name += "_DIAGCRIT"
    if not args.summarize:
        run_name += "_NOSUMMARY"
    if args.run_descr:
        run_name += str(args.run_descr)
    run_dir = join(args.local_logging_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # Setup logfile and logpickle
    results_log_path = join(run_dir, f"{run_name}_results.pkl")
    eval_log_path = join(run_dir, f"{run_name}_eval.pkl")
    log_path = join(run_dir, f"{run_name}.log")
    logger.add(log_path, enqueue=True, backtrace=True, diagnose=True)
    langchain.debug = True
    for warning in CLI_ADAPTATION_WARNINGS:
        logger.warning(warning)
    CLI_ADAPTATION_WARNINGS.clear()

    # Set langsmith project name
    # os.environ["LANGCHAIN_PROJECT"] = run_name

    # Predict for all patients
    first_patient_seen = False
    for _id in hadm_info_clean.keys():
        if args.first_patient and not first_patient_seen:
            if _id == args.first_patient:
                first_patient_seen = True
            else:
                continue

        logger.info(f"Processing patient: {_id}")

        # Build
        agent_executor = build_agent_executor_ZeroShot(
            patient=hadm_info_clean[_id],
            llm=llm,
            lab_test_mapping_path=args.lab_test_mapping_path,
            logfile=log_path,
            max_context_length=args.max_context_length,
            tags=tags,
            include_ref_range=args.include_ref_range,
            bin_lab_results=args.bin_lab_results,
            include_tool_use_examples=args.include_tool_use_examples,
            provide_diagnostic_criteria=args.provide_diagnostic_criteria,
            summarize=args.summarize,
            model_stop_words=args.stop_words,
        )

        # Run
        result = agent_executor(
            {"input": hadm_info_clean[_id]["Patient History"].strip()}
        )
        append_to_pickle_file(results_log_path, {_id: result})

if __name__ == "__main__":
    _adapt_slurm_cli_args()
    run()
