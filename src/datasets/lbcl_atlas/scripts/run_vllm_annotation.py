"""
CLI wrapper for vLLM annotation generation.
Can be run standalone with uv: uv run python scripts/run_vllm_annotation.py --help

Uses openai/gpt-oss-120b with Harmony format.
"""
import argparse
import asyncio
import json
import logging
import sys
import time
import atexit
from pathlib import Path

import pandas as pd
import yaml
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_output,
    get_user_message,
    render_for_completion,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate NL annotations via vLLM")
    p.add_argument("--split-yaml", required=True, help="YAML file with sample metadata")
    p.add_argument("--prompt-file", required=True, help="Prompt template file")
    p.add_argument("--few-shot-dir", required=True, help="Directory with {0-8}_{request,response}.json")
    p.add_argument("--output-csv", required=True, help="Output CSV path")
    p.add_argument("--log-file", default=None, help="Log file path")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="Model path")
    p.add_argument("--quantization", default=None, help="Quantization method")
    p.add_argument("--max-num-seqs", type=int, default=16, help="Max concurrent sequences")
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism (num GPUs)")
    p.add_argument("--study-specific-fields", nargs="+",
                   default=["study_description", "study_title"],
                   help="Fields to treat as study-level context")
    return p.parse_args()


args = parse_args()

STUDY_SPECIFIC_FIELDS = args.study_specific_fields

# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
if args.log_file:
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Load inputs
prompt_template = Path(args.prompt_file).read_text()

with open(args.split_yaml) as f:
    yaml_split = yaml.load(f, Loader=yaml.FullLoader)

# Build few-shot block
few_shot_dir = Path(args.few_shot_dir)
few_shot_block = []
for i in range(9):
    req_file = few_shot_dir / f"{i}_request.json"
    resp_file = few_shot_dir / f"{i}_response.json"
    if not req_file.exists():
        break
    data = json.loads(req_file.read_text())
    sample_data = {k: v for k, v in data.items() if k not in STUDY_SPECIFIC_FIELDS}
    study_data = {k: v for k, v in data.items() if k in STUDY_SPECIFIC_FIELDS}
    example = (
        f"Study Information: {json.dumps(study_data)}\n"
        f"Sample Information: {json.dumps(sample_data)}"
        f"\nResponse: {resp_file.read_text()}\n"
    )
    few_shot_block.append(example)

logger.info(f"Processing {len(yaml_split)} samples with model {args.model}")
logger.info(f"Loaded {len(few_shot_block)} few-shot examples")

# Initialize vLLM engine
engine_args = AsyncEngineArgs(
    model=args.model,
    quantization=args.quantization,
    dtype="auto",
    gpu_memory_utilization=0.90,
    max_num_seqs=args.max_num_seqs,
    tensor_parallel_size=args.tensor_parallel_size,
    enable_prefix_caching=True,
    disable_log_stats=False,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
engine_running = True


def cleanup_handler():
    global engine_running
    if engine_running:
        logger.info("Emergency cleanup: attempting to shutdown engine...")


atexit.register(cleanup_handler)

sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)


def build_query(sample):
    sample_data = {k: v for k, v in sample.items() if k not in STUDY_SPECIFIC_FIELDS}
    study_data = {k: v for k, v in sample.items() if k in STUDY_SPECIFIC_FIELDS}
    return (
        f"Study Information: {json.dumps(study_data)}\n"
        f"Sample Information: {json.dumps(sample_data)}"
    )


def extract_final_output(llm_output) -> str:
    output_token_ids = llm_output.outputs[0].token_ids
    reasoning, content, _ = parse_chat_output(output_token_ids)
    if reasoning:
        logger.debug(f"Extracted reasoning: {repr(reasoning[:100])}")
    if content:
        logger.debug(f"Extracted content: {repr(content[:100])}")
    if content is not None:
        return content.strip()
    logger.warning("Could not parse harmony format, using raw text output")
    return llm_output.outputs[0].text.strip()


async def process_single_sample(key, sample):
    query = build_query(sample)
    if len(query) > 80000:
        query = query[:80000]

    prompt = prompt_template.format(
        few_shot_block="\n".join(few_shot_block), query=query
    )

    user_msg = get_user_message(prompt)
    token_ids = render_for_completion([user_msg])

    request_id = random_uuid()
    results_generator = engine.generate(
        {"prompt_token_ids": token_ids}, sampling_params, request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        logger.error(f"No output for {key}")
        return key, "No information available for this sample"

    raw_text = final_output.outputs[0].text.strip()
    logger.debug(f"{key} raw response: {repr(raw_text[:200])}")

    annotation = extract_final_output(final_output)

    # Try to parse JSON (chain-of-thought format from prompt)
    try:
        result = json.loads(annotation)
        annotation = result.get("2. Final Response", annotation)
    except json.JSONDecodeError:
        pass

    if len(annotation) < 20:
        logger.warning(f"Short annotation for {key}: {annotation}")
        annotation = "No information available for this sample"

    logger.info(f"Processed {key}: {annotation[:100]}...")
    return key, annotation


async def main():
    try:
        tasks = [process_single_sample(k, v) for k, v in yaml_split.items()]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        rows = [{"sample_id": k, "replicate": 0, "annotation": ann} for k, ann in results]
        df = pd.DataFrame(rows)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

        logger.info(f"Processed {len(yaml_split)} samples in {elapsed:.1f}s")
    finally:
        global engine_running
        logger.info("Shutting down vLLM engine...")
        try:
            if hasattr(engine, 'shutdown'):
                await engine.shutdown()
            elif hasattr(engine, 'stop_remote_worker_execution_loop'):
                await engine.stop_remote_worker_execution_loop()
            engine_running = False
        except Exception as e:
            logger.warning(f"Error during engine shutdown: {e}")
        logger.info("Engine shutdown complete")


asyncio.run(main())
