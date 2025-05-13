> Most importantly, if this exact same input is provided to any LLM, such as GPT-4 or Claude-v1-Sonnet as well as the original Mistral 7B, without any fine-tuning, it would be able to provide very reasonable responses about that specific cell type. The authors should clearly quantify what benefit the CellWhisperer transcriptome embedding or LLM fine-tuning is offering here over an off-the-shelf LLM with the same hidden prompt listing the top 50 expressed genes. 



## Cost for different methods (log scale)

Estimated A100 cost: 1 USD/hour

### GPT-4o

Cost per token (input): $3.750 / 1M tokens
Cost per token (output): $15

Number of tokens: ~ 250

Cost per 1M cells: 250 * 3.75 USD = 937.5

### Antrophic Claude

Cost per token (input): $3 / 1M tokens
Cost per token (output): $15

Number of tokens: ~ 250

Cost per 1M cells: 250 * 3 USD = 750

### Llama 3.3 70B (ran locally)

~10 hours runtime for ~15000 cells

1000000cells * (10h / 15000cells) = 667 hours for 1M cells

Cost for 1M cells: 667 USD

### Mistral 7B (generative. estimated based on llama 3.3 1/10th parameter count)

~1 hour runtime for ~15000 cells

1000000cells * (1h / 15000cells) = 66.7 hours for 1M cells

Cost for 1M cells: 66.7 USD

### Mistral 7B (with top 50 genes)

1000000 * 0.2 / 15000 =  13.3

Cost for 1M cells: 13.3 USD

### Mistral 7B (with embedding, i.e. CellWhisperer chat model)

1000000 * 0.08 / 15000 =  5.33

Cost for 1M cells: 5.3 USD + 0.7 USD (for Geneformer embedding)

### CellWhisperer embedding model

CW-Geneformer: 0.1h
CW-UCE: 0.5h
CW-scGPT: 0.05h

CW-Geneformer: 0.1* 1000000 * (0.1 / 15000) = 0.667 USD
CW-UCE: 0.1 * 1000000 * (0.5 / 15000) = 3.33 USD
CW-scGPT: 0.1 * 1000000 * (0.05 / 15000) = 0.33 USD


### CellWhisperer embedding model (precomputed embeddings)

0.001h = 3.6 seconds

Cost: 0.0001 USD
