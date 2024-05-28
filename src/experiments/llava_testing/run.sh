
conda activate llava
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-path ~/cellwhisperer/results/llava/finetuned/Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1/ &
python -m llava.serve.controller --host 0.0.0.0 --port 10000 2>&1 > controller.log &
conda activate cellwhisperer
make start
