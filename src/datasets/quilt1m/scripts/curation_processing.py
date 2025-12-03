import asyncio
import json
import logging
from pathlib import Path
import pandas as pd
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
from vllm.entrypoints.harmony_utils import (
    parse_chat_output,
    get_system_message,
    get_user_message,
    render_for_completion,
)

import time
import sys
import signal
import atexit

# Configure logging to write to both file and stdout/stderr
log_file = snakemake.log[0] if hasattr(snakemake, "log") and snakemake.log else None

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG to see parsing details

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler (stdout/stderr)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler if log file is specified
if log_file:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Model configuration from snakemake params
MODEL_PATH = snakemake.params.model_path
QUANTIZATION = snakemake.params.quantization
DTYPE = "auto"
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_SEQS = snakemake.params.max_num_seqs

# Read the prompt file
with open(snakemake.input.prompt_file, "r") as f:
    system_prompt = f.read()

# Read the CSV split to process
df = pd.read_csv(snakemake.input.split_csv)

logger.info(f"Processing {len(df)} rows")

# Initialize vLLM engine
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    quantization=QUANTIZATION,
    dtype=DTYPE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_num_seqs=MAX_NUM_SEQS,
    enable_prefix_caching=True,
    disable_log_stats=False,
)

logger.info(f"Initializing vLLM engine for {MODEL_PATH}")
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Global variable to track if engine is running for cleanup
engine_running = True

def cleanup_handler():
    """Cleanup function for atexit"""
    global engine_running
    if engine_running:
        logger.info("Emergency cleanup: attempting to shutdown engine...")

# Register cleanup function
atexit.register(cleanup_handler)

# Sampling parameters for deterministic output
sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)


def extract_final_output(llm_output) -> str:
    """
    Extract final output from gpt-oss reasoning format using vLLM's built-in parser.

    The gpt-oss model generates output with hidden tokens that separate reasoning
    and final content. We use vLLM's harmony_utils to parse this properly.

    Args:
        llm_output: The LLM output object from vLLM

    Returns:
        The final content from the final channel
    """
    # Get the raw token IDs from the output
    output_token_ids = llm_output.outputs[0].token_ids

    # Use vLLM's built-in harmony parser to extract reasoning and content
    reasoning, content, _ = parse_chat_output(output_token_ids)

    # Debug logging to show what was extracted
    if reasoning:
        logger.debug(f"Extracted reasoning: {repr(reasoning[:100])}")
    if content:
        logger.debug(f"Extracted content: {repr(content[:100])}")

    # We want the final content, not the reasoning
    if content is not None:
        return content.strip()

    # Fallback: if parsing fails, use the raw text
    logger.warning("Could not parse harmony format, using raw text output")
    return llm_output.outputs[0].text.strip()


async def process_single_row(row_idx: int) -> tuple:
    """Process a single CSV row with LLM curation"""
    row = df.iloc[row_idx]

    # Format the row data as JSON for the LLM
    input_data = {
        "caption": row.get("caption", ""),
        "pathology": row.get("pathology", ""),
        "roi_text": row.get("roi_text", ""),
        "med_uml_ids": row.get("med_uml_ids", ""),
        "full_text": row.get(
            "corrected_text", ""
        ),  # Rename corrected_text to full_text
    }

    # Convert to JSON string for the prompt
    input_json = json.dumps(input_data, indent=2)

    # Build Harmony conversation and render to tokens
    # sys_msg = get_system_message(instructions=system_prompt)  # NOTE this fails for some reason
    # print(sys_msg)
    user_msg = get_user_message(
        f"Instructions:\n{system_prompt}\nInput:\n{input_json}\nOutput:"
    )
    token_ids = render_for_completion([user_msg])

    # Generate request with token IDs to ensure Harmony formatting
    request_id = random_uuid()
    results_generator = engine.generate(
        {"prompt_token_ids": token_ids}, sampling_params, request_id
    )

    # Get the result
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        logger.error(f"No output received for row {row_idx}")
        return row_idx, row.get("caption", "")

    # Extract final output from gpt-oss reasoning format using vLLM's parser
    curated_caption = extract_final_output(final_output)

    # Debug logging to understand the format
    raw_text = final_output.outputs[0].text.strip()
    logger.debug(f"Row {row_idx} raw response: {repr(raw_text[:200])}")
    logger.debug(f"Row {row_idx} extracted final: {repr(curated_caption[:100])}")

    logger.info(
        f"Processed row {row_idx}: '{row.get('caption', '')}' -> '{curated_caption[:100]}...'"
    )
    return row_idx, curated_caption


async def main():
    """Main async function to process all rows"""
    try:
        # Create tasks for all rows
        tasks = [process_single_row(i) for i in range(len(df))]

        # Process concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Update the dataframe with curated captions
        df_curated = df.copy()
        for row_idx, curated_caption in results:
            df_curated.at[row_idx, "caption"] = curated_caption

        # Save the curated CSV
        Path(snakemake.output.processed_csv).parent.mkdir(parents=True, exist_ok=True)
        df_curated.to_csv(snakemake.output.processed_csv, index=False)

        logger.info(f"Processed {len(df)} rows in {end_time - start_time:.2f} seconds")
        logger.info(f"Saved curated metadata to {snakemake.output.processed_csv}")
    
    finally:
        # Properly shutdown the vLLM engine to prevent cleanup errors
        global engine_running
        logger.info("Shutting down vLLM engine...")
        try:
            # Check if engine has shutdown method
            if hasattr(engine, 'shutdown'):
                await engine.shutdown()
            elif hasattr(engine, 'stop_remote_worker_execution_loop'):
                await engine.stop_remote_worker_execution_loop()
            engine_running = False
        except Exception as e:
            logger.warning(f"Error during engine shutdown: {e}")
        
        logger.info("Engine shutdown complete")


# Run the main function
asyncio.run(main())
