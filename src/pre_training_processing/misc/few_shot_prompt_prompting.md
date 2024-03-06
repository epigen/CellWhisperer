## Generation of few-shot prompt
SYSTEM: Create an effective few-shot prompt for GPT-3.5, based on the provided instruction and examples

HUMAN:

General Instruction:
You are given a structured (JSON) annotation of a RNA sequencing study with detailed information about a single sample from this study. Your job is to formulate a short and concise formulation of this single sample in natural language. 

Take special attention to the following points:

- The JSON information provides context about the study in which the cellular context of interest was observed. This context may provide predominantly irrelevant information with respect to the cell state, so pay special attention to the sample-specific information in the JSON.
- Provide only information that is relevant to the cell state (e.g. cell type, perturbation, ...) in any manner. In other words, focus on biological properties, which are reflected in the cellular phenotype and transcriptome of the sample.
- Disregard information that is not reflected in the phenotype or transcriptome of the sample. E.g., discard all study-specific information.




Example Input 1:
        "sra_title": "1158_PFC",
        "sra_study_title": "Homo sapiens Transcriptome or Gene expression",
        "sra_study_abstract": "Despite its tremendous heterogeneity, autism is characterized by a shared core behavioral phenotype. This suggests a convergence of the pathology on specific neural substrates. We aimed to identify region-specific molecular changes by performing the largest to date meta analysis of RNA-seq data from six cortical regions of autism patients together with DNA methylation profiling of one of the regions. We identified a prominent discordant pattern of gene expression and splicing changes that suggests a hyperactivation of the prefrontal cortex and a hypoactivation of the anterior insular cortex involved in emotional responses. Moreover, we identified a link between DNA methylation, alternative splicing and gene expression changes mediated by the chromatin modifier and high autism risk factor gene CHD8. Finally, analysis of candidate gene expression in different regions of developing human cortex suggested a predominant enrichment of misregulated genes in the layers populated by radial glia cells and producing inhibitory interneurons.",
        "sra_sample_attributes": "isolate=1158;age=16;biomaterial_provider=UMB Brain & Tissue Bank;sex=male;tissue=prefrontal cortex;BioSampleModel=Human",
        "sample_name": NaN,
        "mapped_ontology_terms": NaN,
        "raw_SRA_metadata": NaN,
        "geo_title": NaN,
        "Tissue": "Brain",
        "Tissue_subtype": "Prefrontal cortex",
        "Disease": "Healthy",
        "Disease_subtype": NaN,
        "Treated": 0,
        "Type": "Primary",
        "Comment": NaN


Example Output 1:
The sample is derived from the prefrontal cortex tissue of a 16-year-old male human, provided by the UMB Brain & Tissue Bank. The individual was healthy at the time of sample collection. The sample is primary and has not undergone any treatment. The study aims to identify region-specific molecular changes in autism patients, but this particular sample does not pertain to the disease condition.


Example Input 2:
        "sra_title": "Illumina HiSeq 2000 paired end sequencing; Low grade gliomas subtype analysis",
        "sra_study_title": "Low grade gliomas subtype analysis",
        "sra_study_abstract": "Low grade gliomas (LGG; WHO grade 2 astrocytomas, oligodendrogliomas and oligoastrocytomas) account for about 25% of diffuse gliomas. Most occur in young adults between the ages of 30 and 45 years, and are usually only diagnosed after a seizure. In general, they can be characterised by a long period of continuous slow growth, followed by malignant transformation that will be the cause of death up to 25 years after onset. However, there is a significant number of patients for whom malignant progression is more rapid, with mortality observed within 5 years. This suggests that, as with other tumour types, there may be different subtypes of LGG with specific prognosis. It follows that being able to identify these subtypes may permit better patient stratification and aid targeted treatments. Until recently, our understanding of the variables involved in patient prognosis included the type of tumour \u00e2?? oligodendroglial tumours indicate better prognosis than oligoastrocytic or astrocytic \u00e2?? and presence of the 1p-19q co-deletion. In addition, the recent discovery of mutations in IDH1&2 in the majority of LGGs provided another means of stratifying patients, while offering an important insight into their biology. However, we still understand very little of the biology behind the genesis and progression of the 70-80% of LGG that bear IDH1&2 mutations, let alone the remaining IDH wild-type tumours.",
        "sra_sample_attributes": "ENA first public=2015-07-06;ENA last update=2018-03-09;External Id=SAMEA3474043;INSDC center alias=School of Computer Sciences University of Birmingham;INSDC center name=School of Computer Sciences University of Birmingham;INSDC first public=2015-07-06T17:01:43Z;INSDC last update=2018-03-09T02:14:51Z;INSDC status=public;Submitter Id=E-MTAB-3708:Sample 19;age=32;broker name=ArrayExpress;common name=human;disease=astrocytoma;disease staging=Grade 2;sample name=E-MTAB-3708:Sample 19;sex=male",
        "sample_name": NaN,
        "mapped_ontology_terms": "astrocytoma, male organism",
        "raw_SRA_metadata": "Alias: E-MTAB-3708; Broker name: ArrayExpress; Description: Protocols; ENA checklist: ERC000011; INSDC center alias: School of Computer Sciences University of Birmingham; INSDC center name: School of Computer Sciences University of Birmingham; INSDC first public: 2015-07-06T17; INSDC last update: 2018-03-09T02; INSDC status: public; SRA accession: ERS785319; Sample Name: ERS785319; Title: Sample 19; age: 32; disease: astrocytoma; disease staging: Grade 2; organism: Homo sapiens; sex: male",
        "geo_title": NaN,
        "Tissue": "Brain",
        "Tissue_subtype": NaN,
        "Disease": "Brain cancer",
        "Disease_subtype": "astrocytoma",
        "Treated": 0,
        "Type": "Primary",
        "Comment": NaN


Example Output 2: 
The sample is from a 32-year-old male patient diagnosed with Grade 2 astrocytoma, a subtype of low-grade gliomas. The tissue sample was taken from the brain, which is the primary site of the disease. The sample was not treated before sequencing. The sequencing was performed using Illumina HiSeq 2000 paired-end sequencing.




Example Input 3:
        "sra_title": "Illumina HiSeq 2500 paired end sequencing; RNA-seq of human skin lesions diagnosed as actinic keratosis, intraepidermal carcinoma or squamous cell carcinoma",
        "sra_study_title": "RNA-seq of human skin lesions diagnosed as actinic keratosis, intraepidermal carcinoma or squamous cell carcinoma",
        "sra_study_abstract": "We report whole tissue RNA-seq data captured from 25 AK, IEC or SCC lesions obtained from the skin of 17 individuals. We used RNA-seq to detect and measure HPV E7 transcription.",
        "sra_sample_attributes": "ENA first public=2018-07-10;ENA last update=2018-01-30;External Id=SAMEA104555226;INSDC center alias=Division of Genomics of Development and Disease, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia, 4072;INSDC center name=Division of Genomics of Development and Disease, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia, 4072;INSDC first public=2018-07-10T17:03:24Z;INSDC last update=2018-01-30T09:38:49Z;INSDC status=public;Submitter Id=E-MTAB-6430:Sample 24;age=81;broker name=ArrayExpress;clinical history=immunosuppressed;common name=human;date collected=21_05_2012;disease=squamous cell carcinoma;individual=PAT016;organism part=skin;sample name=E-MTAB-6430:Sample 24;sampling site=arm;sex=female",
        "sample_name": NaN,
        "mapped_ontology_terms": "squamous cell carcinoma, skin, zone of skin, arm, female organism",
        "raw_SRA_metadata": "Alias: E-MTAB-6430; Broker name: ArrayExpress; Description: Protocols; ENA checklist: ERC000011; INSDC center alias: Division of Genomics of Development and Disease, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia, 4072; INSDC center name: Division of Genomics of Development and Disease, Institute for Molecular Bioscience, The University of Queensland, Brisbane, Queensland, Australia, 4072; INSDC first public: 2018-07-10T17; INSDC last update: 2018-01-30T09; INSDC status: public; SRA accession: ERS2166510; Sample Name: ERS2166510; Title: Sample 24; age: 81; clinical history: immunosuppressed; date collected: 21_05_2012; disease: squamous cell carcinoma; organism: Homo sapiens; organism part: skin; sampling site: arm; sex: female",
        "geo_title": NaN,
        "Tissue": "Skin",
        "Tissue_subtype": NaN,
        "Disease": "Skin cancer",
        "Disease_subtype": "Squamous cell carcinoma",
        "Treated": 0,
        "Type": "Primary",
        "Comment": "All patients very old; study also contains scRNA-seq from mouse models, some immunosuppressed"

Example Output 3:
This sample is from an 81-year-old immunosuppressed female patient diagnosed with squamous cell carcinoma, a subtype of skin cancer. The sample was collected from a lesion on the patient's arm on May 21, 2012. The RNA sequencing was performed to detect and measure HPV E7 transcription. The study did not involve any treatment and the sample is of primary type.
