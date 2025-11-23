# CellWhisperer Validation Registry

This module provides a **DRY (Don't Repeat Yourself) solution** for managing validation benchmarks across both training validation and Snakemake evaluation pipelines.

## 🎯 Problem Solved

Previously, validation benchmarks were defined in multiple places:
- `fig2_embedding_validations.smk` (comprehensive evaluation)
- `initialize_validation_functions` (training validation)
- `cellwhisperer_benchmark.smk` (spotwhisperer evaluation)

This led to code duplication, inconsistencies, and maintenance overhead.

## 🏗️ Solution Architecture

### **Centralized Registry** (`registry.py`)
- **Single source of truth** for all benchmark definitions
- Defines 8 comprehensive benchmarks from fig2 evaluation:
  - Tabula Sapiens (celltype, organ_tissue)
  - Human Disease (disease_subtype, tissue) 
  - Pancreas, ImmGen, AIDA (celltype)
  - Tabula Sapiens well-studied celltypes

### **Dual Access Pattern**

#### **1. Training Validation** (Efficient)
```python
# Lightweight validation (default)
validation_fns = initialize_validation_functions(
    batch_size=32,
    transcriptome_model_type="geneformer",
    text_model_type="bert",
    image_model_type="uni2",
    enable_comprehensive_benchmarks=False  # Fast training
)

# Comprehensive validation (final evaluation)
validation_fns = initialize_validation_functions(
    # ... same params ...
    enable_comprehensive_benchmarks=True  # All benchmarks
)
```

#### **2. Snakemake Evaluation** (Flexible)
```bash
# Run specific comprehensive benchmarks
snakemake cellwhisperer_comprehensive_zero_shot --config dataset=tabula_sapiens metadata_col=celltype

# Run all comprehensive benchmarks
snakemake aggregate_comprehensive_benchmarks

# Include in spider plot (automatic)
snakemake spider_performance_plot
```

## 📊 Integration Points

### **Training Pipeline**
- `cellwhisperer test` now supports comprehensive benchmarks via flag
- Results automatically flow into `aggregated_cwevals.csv`
- Compatible with existing validation infrastructure

### **Snakemake Pipeline** 
- New rule: `cellwhisperer_comprehensive_zero_shot`
- Aggregation: `aggregate_comprehensive_benchmarks`
- **Spider plot integration**: Comprehensive metrics automatically included

### **Spider Plot Enhancement**
The spider plot now includes comprehensive zero-shot metrics:
```python
"text-transcriptome": [
    "valfn_zshot_TabSap_cell_lvl/f1_macroAvg",
    "valfn_zshot_TabSap_cell_lvl/rocauc_macroAvg",
    # NEW: Comprehensive benchmarks
    "zshot_HumanDisease_disease_subtype/f1",
    "zshot_Pancreas_celltype/rocauc", 
    "zshot_ImmGen_celltype/f1",
    "zshot_AIDA_celltype/accuracy",
    "zshot_TabSap_organ_tissue/f1",
]
```

## 🚀 Usage Examples

### **Adding New Benchmarks**
Simply add to `ValidationRegistry.get_cellwhisperer_benchmarks()`:
```python
ValidationBenchmarkSpec(
    name="zshot_MyDataset_celltype",
    dataset="my_dataset",
    metadata_col="celltype",
    dataset_kwargs={"celltype_obs_colname": "cell_type"},
    processor_kwargs={},
    description="My dataset celltype prediction",
    category="celltype"
)
```

### **Selective Evaluation**
```python
# Get benchmarks by category
celltype_benchmarks = ValidationRegistry.get_benchmarks_by_category("celltype")
disease_benchmarks = ValidationRegistry.get_benchmarks_by_category("disease") 

# Get specific benchmark
tabsap_spec = ValidationRegistry.get_benchmark_by_name("zshot_TabSap_celltype")
```

### **Training with Comprehensive Validation**
```python
# In your training script
validation_fns = initialize_validation_functions(
    batch_size=32,
    transcriptome_model_type="geneformer", 
    text_model_type="bert",
    image_model_type="uni2",
    enable_comprehensive_benchmarks=True  # Enable all fig2 benchmarks
)

# All 8 comprehensive benchmarks will be available
# Results saved to metrics.csv with consistent naming
```

## 📈 Benefits

✅ **DRY Compliance**: Single source of truth for all validations
✅ **Efficiency**: Lazy loading for training, full flexibility for evaluation  
✅ **Consistency**: Same metrics across training and evaluation
✅ **Flexibility**: Easy to add/modify benchmarks
✅ **Integration**: Seamless spider plot enhancement
✅ **Backward Compatibility**: Existing code continues to work

## 🔧 Files Created/Modified

### **New Files**
- `src/cellwhisperer/validation/registry.py` - Central benchmark registry
- `src/spotwhisperer_eval/rules/cellwhisperer_comprehensive_benchmark.smk` - Snakemake rules
- `src/spotwhisperer_eval/notebooks/cellwhisperer_comprehensive_zero_shot.py.ipynb` - Evaluation notebook  
- `src/spotwhisperer_eval/scripts/aggregate_comprehensive_benchmarks.py` - Aggregation script

### **Modified Files**
- `src/cellwhisperer/validation/__init__.py` - Added lazy loading support
- `src/spotwhisperer_eval/Snakefile` - Included comprehensive benchmark rules
- `src/spotwhisperer_eval/rules/plots.smk` - Enhanced spider plot with comprehensive metrics

## 🎯 Migration Path

1. **Immediate**: Use new registry for future benchmarks
2. **Gradual**: Migrate existing fig2 evaluations to use registry
3. **Future**: Deprecate old duplicate validation definitions

This architecture provides the **efficiency of selective validation during training** AND the **flexibility of comprehensive Snakemake-based evaluation**, while maintaining DRY principles through a centralized registry.