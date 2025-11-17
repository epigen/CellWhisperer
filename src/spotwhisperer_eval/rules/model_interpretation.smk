# Model interpretation and disease analysis workflow
# Analyzes disease detectability and performance improvements using LLM-based scoring

# Results paths for model interpretation
MODEL_INTERPRETATION_RESULTS = PROJECT_DIR / "results/spotwhisperer_eval/benchmarks/model_interpretation"

rule llm_histopathology_analysis:
    """
    Score disease detectability from H&E text using an LLM; outputs classifications, analysis, and plots.
    """
    input:
        per_class_analysis=rules.cellwhisperer_per_class_analysis.output.analysis,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        classifications=MODEL_INTERPRETATION_RESULTS / "disease_histopathology_classifications.json",
        analysis_csv=MODEL_INTERPRETATION_RESULTS / "histopathology_detectability_analysis.csv",
        plots=report(MODEL_INTERPRETATION_RESULTS / "histopathology_detectability_analysis.png", 
                    category="model_interpretation", subcategory="llm_analysis", 
                    labels={"Analysis": "H&E Detectability", "Format": "plot"})
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/llm_based_analysis.py"

rule llm_hest_representation_analysis:
    """
    Assess likely disease representation in HEST via LLM; outputs classifications, description, analysis, and plots.
    """
    input:
        histopathology_classifications=rules.llm_histopathology_analysis.output.classifications,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        hest_classifications=MODEL_INTERPRETATION_RESULTS / "disease_hest_representation_classifications.json",
        hest_description=MODEL_INTERPRETATION_RESULTS / "hest_dataset_description.txt",
        analysis_csv=MODEL_INTERPRETATION_RESULTS / "hest_representation_analysis.csv",
        plots=report(MODEL_INTERPRETATION_RESULTS / "hest_representation_analysis.png",
                    category="model_interpretation", subcategory="llm_analysis",
                    labels={"Analysis": "HEST Representation", "Format": "plot"})
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/llm_hest1k_disease.py"

rule correlation_analysis:
    """
    Correlate LLM detectability and representation scores with performance changes.
    """
    input:
        per_class_analysis=rules.cellwhisperer_per_class_analysis.output.analysis,
        histopathology_classifications=rules.llm_histopathology_analysis.output.classifications,
        hest_classifications=rules.llm_hest_representation_analysis.output.hest_classifications,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        analysis_data=MODEL_INTERPRETATION_RESULTS / "correlation_analysis_data.csv",
        correlation_plots=report(MODEL_INTERPRETATION_RESULTS / "correlation_he_detectability_vs_performance.png",
                               category="model_interpretation", subcategory="correlation_analysis",
                               labels={"Analysis": "H&E Detectability Correlation", "Format": "plot"}),
        four_group_plots=report(MODEL_INTERPRETATION_RESULTS / "four_group_box_plots.png",
                              category="model_interpretation", subcategory="correlation_analysis",
                              labels={"Analysis": "Four Group Comparison", "Format": "plot"})
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/correlation_analysis.py"

rule high_detectability_disease_analysis:
    """
    Analyze high-detectability diseases grouped by F1 changes; output CSVs and plots.
    """
    input:
        hest_representation_analysis=rules.llm_hest_representation_analysis.output.analysis_csv,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        full_analysis=MODEL_INTERPRETATION_RESULTS / "high_detectability_full_analysis.csv",
        category_summary=MODEL_INTERPRETATION_RESULTS / "high_detectability_category_summary.csv",
        comprehensive_plots=report(MODEL_INTERPRETATION_RESULTS / "high_detectability_comprehensive_analysis.png",
                                 category="model_interpretation", subcategory="high_detectability",
                                 labels={"Analysis": "High Detectability Comprehensive", "Format": "plot"}),
        comprehensive_plots_svg=MODEL_INTERPRETATION_RESULTS / "high_detectability_comprehensive_analysis.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=30000,
        slurm="cpus-per-task=4"
    script:
        "../scripts/high_detectability_disease_analysis.py"

rule distribution_comparison_analysis:
    """
    Compare detectability score distributions for top-improving diseases; output plots and top lists.
    """
    input:
        per_class_analysis=rules.cellwhisperer_per_class_analysis.output.analysis,
        hest_representation_analysis=rules.llm_hest_representation_analysis.output.analysis_csv,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        distribution_plots=report(MODEL_INTERPRETATION_RESULTS / "distribution_comparison_detectability_scores.png",
                                category="model_interpretation", subcategory="distribution_analysis",
                                labels={"Analysis": "Distribution Comparison", "Format": "plot"}),
        detectability_violin_plots=report(MODEL_INTERPRETATION_RESULTS / "detectability_comparison_violin_plots.png",
                                        category="model_interpretation", subcategory="distribution_analysis",
                                        labels={"Analysis": "Detectability Violin Plots", "Format": "plot"}),
        detectability_violin_svg=MODEL_INTERPRETATION_RESULTS / "detectability_comparison_violin_plots.svg",
        top_f1_diseases=MODEL_INTERPRETATION_RESULTS / "top_15_f1_improving_diseases.csv",
        top_rocauc_diseases=MODEL_INTERPRETATION_RESULTS / "top_15_rocauc_improving_diseases.csv"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/distribution_comparison_detectability_scores.py"

rule quilt1m_mention_correlation:
    """
    Correlate Quilt1M disease mentions with performance changes; output data and plot.
    """
    input:
        per_class_analysis=rules.cellwhisperer_per_class_analysis.output.analysis,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        correlation_data=MODEL_INTERPRETATION_RESULTS / "quilt1m_mention_correlation_data.csv",
        correlation_plots=report(MODEL_INTERPRETATION_RESULTS / "quilt1m_mention_correlation_analysis.png",
                               category="model_interpretation", subcategory="quilt1m_analysis",
                               labels={"Analysis": "Quilt1M Mention Correlation", "Format": "plot"})
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/quilt1m_mention_correlation_analysis.py"

rule adhoc_per_class_analysis:
    """
    Exploratory plots for human_disease per-class results.
    """
    input:
        per_class_analysis=rules.cellwhisperer_per_class_analysis.output.analysis,
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        analysis_plots=report(MODEL_INTERPRETATION_RESULTS / "human_disease_analysis.png",
                            category="model_interpretation", subcategory="adhoc_analysis",
                            labels={"Analysis": "Human Disease Analysis", "Format": "plot"}),
        mention_correlation_plots=MODEL_INTERPRETATION_RESULTS / "human_disease_mention_correlation.png",
        violin_plots=MODEL_INTERPRETATION_RESULTS / "human_disease_violin_plot.png"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    script:
        "../scripts/adhoc_per_class_analysis.py"

rule model_interpretation_all:
    """
    Run all model interpretation analyses end-to-end.
    """
    input:
        # LLM-based analyses
        rules.llm_histopathology_analysis.output.plots,
        rules.llm_hest_representation_analysis.output.plots,
        
        # Correlation and comparison analyses
        rules.correlation_analysis.output.four_group_plots,
        rules.high_detectability_disease_analysis.output.comprehensive_plots,
        rules.distribution_comparison_analysis.output.detectability_violin_plots,
        
        # Additional analyses
        rules.quilt1m_mention_correlation.output.correlation_plots,
        rules.adhoc_per_class_analysis.output.analysis_plots,
