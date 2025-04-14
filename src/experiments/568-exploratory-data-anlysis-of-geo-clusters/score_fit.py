import re
import yaml
import json
from llama_cpp import Llama, LlamaGrammar

def parse_yaml_cells(yaml_content):
    """Parse the YAML file and extract individual cell descriptions."""
    # Extract the title and cells list
    match = re.match(r'([^:]+):([\s\S]+)', yaml_content)
    if not match:
        raise ValueError("Invalid YAML format")
    
    title = match.group(1).strip()
    cells_text = match.group(2).strip()
    
    # Split by dash markers, which indicate new cell entries
    cells = []
    for cell in re.split(r'\n- ', cells_text):
        if cell.startswith('- '):  # Handle the first item which might have the dash
            cell = cell[2:]
        cell = cell.strip()
        if cell:
            # Replace newlines within a cell with spaces
            cell = re.sub(r'\n\s+', ' ', cell)
            cells.append(cell)
    
    return title, cells

# def create_scoring_prompt(cluster_label, cell_description):
#     """Create a prompt for scoring a cell against a cluster label with few-shot examples."""
#     return f"""<s>[INST] I have a cluster labeled "{cluster_label}". Please score the given cell metadata based on how closely it aligns with the cluster label, considering factors like cell type, developmental stage, tissue origin, disease, donor information and experimental context.
# Score from **0 to 10**, where:
# * **10** means the sample is highly representative of the cluster label (i.e., the metadata description strongly reflects the characteristics described by the cluster label).
# * **0** means the sample is unrelated or irrelevant to the cluster label (i.e., the metadata description does not match the general theme of the label).
# * Scores in between should reflect the degree of relevance, based on the similarity of the sample's characteristics to the cluster label.
# Here are some example scorings for a cluster labeled "Hippocampal Neurons from 65-70 Year Old Alzheimer's Patients":

# Example 1: "Mature hippocampal neurons isolated from post-mortem brain tissue of a 67-year-old female with late-stage Alzheimer's disease (Braak stage V-VI), showing amyloid plaques and tau tangles, cultured for 24 hours in glucose-deprived medium to simulate metabolic stress."
# Score: 10 (Highly relevant as these are hippocampal neurons from a patient within the exact age range, with confirmed Alzheimer's pathology)

# Example 2: "Hippocampal neural progenitor cells derived from a 68-year-old male donor with mild cognitive impairment but no confirmed Alzheimer's diagnosis, maintained in neural differentiation medium for 14 days."
# Score: 7 (Partially relevant as these are hippocampal cells within the specified age range, but they are progenitors rather than mature neurons, and critically, from a pre-Alzheimer's state without confirmed disease)

# Example 3: "Cortical neurons isolated from temporal lobe of a 66-year-old female with late-onset Alzheimer's disease, expressing ApoE4 genotype, treated with amyloid-beta oligomers for 48 hours."
# Score: 8 (Mostly relevant as these are mature neurons from an Alzheimer's patient within the age range, with the main difference being cortical rather than hippocampal origin)

# Example 4: "Hippocampal neurons extracted from a 42-year-old male with early-onset familial Alzheimer's disease (PSEN1 mutation), showing significant amyloid pathology, cultured in standard neuronal medium."
# Score: 9 (Moderately relevant as these are hippocampal neurons from an Alzheimer's patient with appropriate pathology; the age does not match the specified range)

# Example 5: "Hippocampal tissue samples from a 69-year-old female healthy donor with no cognitive impairment or neuropathology, processed for single-nucleus RNA sequencing."
# Score: 7 (Partially relevant as despite being hippocampal cells within the age range, the absence of Alzheimer's disease is a critical mismatch with the cluster label)

# Example 6: "Peripheral blood monocytes from a 67-year-old male with Alzheimer's disease, isolated and differentiated into microglia-like cells in vitro, co-cultured with neuronal amyloid."
# Score: 2 (Very low relevance as these are not neurons despite being from an Alzheimer's patient within the specified age range; cell type is a fundamental mismatch)

# Now, please score the following cell metadata:
# Cell metadata to score: "{cell_description}"
# Return only a single integer score between 0 and 10. [/INST]"""

# def create_scoring_prompt(cluster_label, cell_description):
#     """Create a prompt for scoring how well a cluster label captures cell metadata.
#     Intended for mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf."""
#     return f"""<s>[INST] I have a cluster labeled "{cluster_label}". Please score how well the cluster label captures the characteristics described in the following cell metadata. Consider the following aspects:
# - Cell type
# - Developmental stage
# - Tissue origin
# - Disease state
# - Donor information
# - Experimental context

# Score from 0 to 10, where:
# * 10 indicates that the cluster label perfectly captures the metadata with no contradictions (even if the label is somewhat general).
# * 0 indicates that the cluster label contradicts the metadata in key aspects.
# * Scores in between reflect partial alignment, with generous scoring for overall similarity even if minor details differ.

# Here are some example scorings for a cluster labeled "Adult Normoxic Cardiac Myocytes from Healthy Donors":
# Example A1: "Cardiac myocytes isolated from adult heart tissue of a 55-year-old healthy male donor, maintained in normoxic culture."  
# Score: 10 (Perfect alignment in every aspect.)
# Example A2: "Cardiac progenitor cells derived from iPSCs of a 45-year-old healthy donor, cultured under hypoxic conditions."  
# Score: 7 (Cardiac lineage and donor information are strong matches, with minor deviations in developmental stage and culture conditions.)
# Example A3: "Cardiac myocytes isolated from adult heart tissue of a 60-year-old patient with a history of myocardial infarction, maintained in normoxic culture."  
# Score: 6 (Good match on cell type, developmental stage, tissue origin, and experimental conditions, but a contradiction in donor health.)
# Example A4: "Vascular smooth muscle cells isolated from adult aorta of a 50-year-old healthy donor, maintained in normoxic culture."  
# Score: 2 (Despite matching donor and culture conditions, the cell type is fundamentally different.)


# Below are some example scorings for a cluster labeled "Developing Midbrain Dopaminergic Neurons":
# Example B1: "Midbrain dopaminergic neuron precursors derived from human embryonic stem cells at day 35 of differentiation."  
# Score: 10 (Strong alignment with the label, as these are developing dopaminergic neurons from a relevant source and stage.)
# Example B2: "Fetal midbrain tissue sample (16 weeks post-fertilization) containing differentiating dopaminergic neurons."  
# Score: 9 (Highly relevant, though a broader tissue sample means it may contain additional cell types.)
# Example B3: "Mature dopaminergic neurons isolated from adult human midbrain."  
# Score: 6 (Matches the cell type but is fully mature, whereas the cluster label specifies a developing population.)
# Example B4: "Cortical neural progenitors derived from human iPSCs at day 28 of differentiation."  
# Score: 4 (Neural lineage matches, but both the region (cortex vs. midbrain) and specific neuronal subtype are incorrect.)
# Example B5: "PBMCs from a healthy donor."  
# Score: 0 (Completely unrelated; wrong cell type.)

# Now, please score how well the following cell metadata is captured by the cluster label:
# Cell metadata to score: "{cell_description}"
# Return only a single integer score between 0 and 10. [/INST]"""

def create_scoring_prompt(cluster_label, cell_description):
    """Create a prompt for scoring a cell against a cluster label"""
    return f"""<s>[INST] I have a cluster labeled "{cluster_label}". Please score the given cell metadata based on how closely it aligns with the cluster label, considering factors like cell type, developmental stage, tissue origin, disease, donor information (such as age) and experimental context.

Score from **0 to 10**, where:
* **10** means the sample is highly representative of the cluster label (i.e., the metadata description strongly reflects the characteristics described by the cluster label).
* **0** means the sample is unrelated or irrelevant to the cluster label (i.e., the metadata description does not match the general theme of the label).
* Scores in between should reflect the degree of relevance, based on the similarity of the sample's characteristics to the cluster label.

Now, please score the following cell metadata:
Cell metadata to score: "{cell_description}"

Return only a single integer score between 0 and 10. [/INST]"""

# def create_scoring_prompt(cluster_label, cell_description):
#     """Create a prompt for scoring a cell against a cluster label with few-shot examples."""
#     return f"""<s>[INST] I have a cluster labeled "{cluster_label}". Please score the given cell metadata based on how closely it aligns with the cluster label, considering factors like cell type, developmental stage, tissue origin, disease, donor information and experimental context.

# Score from **0 to 10**, where:
# * **10** means the sample is highly representative of the cluster label (i.e., the metadata description strongly reflects the characteristics described by the cluster label).
# * **0** means the sample is unrelated or irrelevant to the cluster label (i.e., the metadata description does not match the general theme of the label).
# * Scores in between should reflect the degree of relevance, based on the similarity of the sample's characteristics to the cluster label.

# Here are some example scorings for a cluster labeled "Normoxic HUVECs Maintaining Endothelial Characteristics":

# Example 1: "Human umbilical vein endothelial cells (HUVEC) with siERK1n2 genotype, cultured cells from endothelium."
# Score: 10 (Highly relevant as these are explicitly HUVECs with endothelial origin and maintained under normoxic conditions)

# Example 2: "Liver sinusoidal endothelial cells (LSEC) treated with N1ICD."
# Score: 6 (Moderately relevant as these are endothelial cells, but from liver rather than umbilical vein, and with N1ICD treatment)

# Example 3: "Human cord-blood endothelial cells from donor"
# Score: 8 (Highly relevant as these are human endothelial cells from a similar developmental source, but not specifically HUVECs)

# Example 4: "Human T cells isolated from peripheral blood of healthy donor"
# Score: 0 (Completely irrelevant as these are immune cells, not endothelial cells)

# Now, please score the following cell metadata:
# Cell metadata to score: "{cell_description}"

# Return only a single integer score between 0 and 10. [/INST]"""

def setup_llm():
    """Initialize the LLM with appropriate settings."""
    # Define a grammar to ensure we get a single numeric score
    score_grammar = """
    root ::= score
    score ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10"
    """
    
    grammar = LlamaGrammar.from_string(score_grammar)
    
    # Initialize Mixtral model
    model = Llama(
        model_path="/msc/home/mschae83/cellwhisperer/resources/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",  # Replace with actual path
        n_ctx=32000,  # The max sequence length to use
        n_threads=5,  # The number of CPU threads to use
        n_threads_batch=25,
        n_gpu_layers=86,  # Load the full model into GPU
    )
    
    return model, grammar

def score_cells(yaml_file_path, cluster_label=None, first_n_cells=None):
    """Score cells from a YAML file against a cluster label."""

    # Read YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()
    
    # Parse cells
    title, cells = parse_yaml_cells(yaml_content)
    
    # Use provided cluster label or default to title from YAML
    if cluster_label is None:
        cluster_label = title
    
    # Setup LLM
    model, grammar = setup_llm()
    
    # Debug: Print the cluster label
    print(f"Scoring cells against cluster label: '{cluster_label}'")
    
    # Score each cell
    scores = {}
    for cell in cells[:first_n_cells] if first_n_cells else cells:
        prompt = create_scoring_prompt(cluster_label, cell)
        
        # Debug: Print abbreviated prompt
        print(f"Scoring cell: {cell[:50]}...")
        
        # Get response from Mixtral with grammar constraint
        response = model.create_completion(
            prompt,
            max_tokens=10,
            temperature=0.2,
            grammar=grammar,
            stop=["</s>"]  # Add explicit stop token for Mixtral
        )
        
        # Extract score
        score_text = response['choices'][0]['text'].strip()
        print(f"Raw response: '{score_text}'")
        
        try:
            score = int(score_text)
            # Add to results
            scores[cell] = score
        except ValueError:
            print(f"Warning: Could not parse score '{score_text}' as integer. Using 0.")
            scores[cell] = 0
    
    return scores

def main():
    import argparse
    
    #parser = argparse.ArgumentParser(description='Score cell descriptions against a cluster label using Mixtral.')
    # parser.add_argument('yaml_file', help='Path to YAML file containing cell descriptions')
    # parser.add_argument('--cluster-label', help='Custom cluster label (default: use title from YAML)')
    # parser.add_argument('--output', help='Output file path for JSON results')
    # parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
   # args = parser.parse_args()

    # For manual testing, uncomment and modify these lines:
    args = argparse.Namespace()
    args.yaml_file = "/msc/home/q56ppene/cellwhisperer/cellwhisperer/src/experiments/568-exploratory-data-anlysis-of-geo-clusters/metadata_terms_1_natural_language_annotation.yaml"
    args.cluster_label = None #"Your Cluster Label"
    args.output = "/msc/home/q56ppene/cellwhisperer/cellwhisperer/src/experiments/568-exploratory-data-anlysis-of-geo-clusters/testoutput_1.yaml"
    args.debug = True
    args.first_n_cells=10

    if args.debug:
        print(f"Using YAML file: {args.yaml_file}")
        print(f"Using cluster label: {args.cluster_label}")
    
    # Score cells
    scores = score_cells(args.yaml_file, args.cluster_label, first_n_cells=args.first_n_cells)
    
    # Format and display results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("Scores:")
        print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()