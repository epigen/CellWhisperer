import asyncio
from typing import List
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
import time

# ================= CONFIGURATION =================
# SELECT YOUR MODEL HERE
# Options: "meta-llama/Llama-4-Scout-MoE" or "Qwen/Qwen3-MoE-A14B"
MODEL_PATH = "meta-llama/Llama-4-Scout-MoE"

# B200 OPTIMIZATION
# 'fp8' is native on Blackwell and offers 2x throughput over bf16.
# If using the larger Qwen3 MoE, you might need 'quantization="gptq_marlin"' (4-bit)
# to fit it on a single 192GB card.
QUANTIZATION = "fp8"
DTYPE = "auto"
GPU_MEMORY_UTILIZATION = 0.95  # Use all available VRAM

# BATCHING SETTINGS
MAX_NUM_SEQS = 256  # High batch size for high throughput
# =================================================


async def run_inference():
    # 1. Initialize the Engine with Prefix Caching Enabled
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        quantization=QUANTIZATION,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        # CRITICAL: Enable Automatic Prefix Caching
        enable_prefix_caching=True,
        # Optimizations for MoE models
        disable_log_stats=False,
    )

    print(f"--- Initializing vLLM on B200 for {MODEL_PATH} ---")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 2. Define the Shared Prefix (The "System Prompt")
    # vLLM will calculate this ONCE and cache it for all subsequent requests.
    system_prompt = (
        "You are an expert data analyst. "
        "Extract the key entities and sentiment from the following text. "
        "Return format: JSON."
    )

    # 3. Simulate "Millions" of samples (Here: 100 dummy samples)
    # In production, stream this from a file/database to avoid RAM spikes.
    sample_inputs = [
        f"Customer review #{i}: The product was okay but shipping was slow."
        for i in range(100)
    ]

    print("--- Starting Batch Inference ---")
    start_time = time.time()

    # Define sampling params (Temperature 0 for deterministic data extraction)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    async def process_request(request_id: str, prompt: str):
        # Combine shared prefix + dynamic input
        full_prompt = f"{system_prompt}\n\nInput: {prompt}\nOutput:"

        results_generator = engine.generate(full_prompt, sampling_params, request_id)

        # vLLM returns a stream; we just want the final output
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return final_output

    # Create tasks for all inputs
    # Note: vLLM manages the internal batching. We can just throw tasks at it.
    tasks = []
    for i, prompt in enumerate(sample_inputs):
        request_id = random_uuid()
        tasks.append(process_request(request_id, prompt))

    # Run concurrently
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # 4. Results
    print(f"--- Processed {len(sample_inputs)} samples in {total_time:.2f}s ---")
    print(f"Throughput: {len(sample_inputs) / total_time:.2f} requests/sec")

    # Print a sample
    print(f"\nSample Output:\n{results[0].outputs[0].text}")


if __name__ == "__main__":
    asyncio.run(run_inference())
