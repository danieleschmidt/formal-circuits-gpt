# Experimental Protocol for LLM-Assisted Hardware Verification Research

## Overview

This document defines the standardized experimental protocol for evaluating novel algorithms in LLM-assisted hardware verification. The protocol ensures reproducible, statistically rigorous experiments that enable fair comparison between different approaches.

## Experimental Design Principles

### 1. Reproducibility Requirements
- **Random Seed Control**: All experiments use fixed random seeds (default: 42)
- **Environment Specification**: Docker containers with pinned dependency versions
- **Hardware Consistency**: Experiments run on standardized hardware configurations
- **Version Control**: All code, data, and results under version control

### 2. Statistical Rigor
- **Sample Size**: Minimum 30 circuits per category for statistical power
- **Significance Testing**: Two-tailed t-tests with α = 0.05
- **Effect Size**: Cohen's d for practical significance assessment
- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Multiple Comparisons**: Bonferroni correction for multiple hypothesis testing

### 3. Experimental Controls
- **Baseline Comparisons**: Standardized baseline implementations
- **Ablation Studies**: Systematic component removal/modification
- **Cross-Validation**: K-fold validation for learning-based components
- **Independent Validation**: Hold-out test sets never used in development

## Benchmark Suite Specification

### Circuit Categories and Distribution
| Category | Count | Complexity Range | Source |
|----------|-------|------------------|---------|
| Arithmetic Basic | 127 | 1.0 - 2.5 | Synthetic + Standard |
| Arithmetic Complex | 89 | 2.5 - 5.0 | Industrial + Synthetic |
| Boolean Logic | 156 | 1.0 - 3.0 | Standard + Generated |
| Sequential Simple | 143 | 2.0 - 4.0 | Textbook + Synthetic |
| Sequential Complex | 98 | 4.0 - 7.0 | Industrial + Research |
| Memory Systems | 76 | 3.0 - 6.0 | Industrial + Academic |
| Communication | 67 | 4.0 - 8.0 | Industrial + Standards |
| Processor Components | 54 | 5.0 - 9.0 | Open Source + Industrial |
| Industrial Scale | 37 | 7.0 - 10.0 | Real Designs (Anonymized) |
| **Total** | **847** | **1.0 - 10.0** | **Mixed Sources** |

### Circuit Selection Criteria
- **Diversity**: Representative of real-world verification challenges
- **Complexity Scaling**: Logarithmic distribution across complexity scores
- **Ground Truth**: Manual verification for accuracy assessment
- **Licensing**: Open source or anonymized with permission
- **Documentation**: Complete specifications and expected properties

## Evaluation Metrics

### Primary Metrics
1. **Property Inference Accuracy**: Percentage of correctly inferred properties
2. **Coverage Score**: Fraction of expected properties discovered
3. **Precision**: Fraction of inferred properties that are correct
4. **Recall**: Fraction of correct properties that were inferred
5. **F1-Score**: Harmonic mean of precision and recall

### Secondary Metrics
1. **Verification Time**: Total time for property inference + verification
2. **Convergence Rate**: Fraction of experiments achieving convergence
3. **Confidence Calibration**: Alignment between predicted and actual accuracy
4. **Scalability**: Performance vs. circuit complexity correlation
5. **Resource Usage**: Memory and computational requirements

### Qualitative Metrics
1. **Property Quality**: Expert assessment of inferred property meaningfulness
2. **Completeness**: Coverage of different property types (safety, liveness, functional)
3. **Interpretability**: Clarity and understandability of generated properties
4. **Robustness**: Performance on edge cases and malformed inputs

## Experimental Procedure

### Phase 1: Environment Setup
```bash
# 1. Environment preparation
docker build -t formal-circuits-research .
docker run -it --name experiment-env formal-circuits-research

# 2. Dependency verification
pip install -r requirements-research.txt
python -m pytest tests/test_environment.py

# 3. Benchmark preparation
python scripts/prepare_benchmarks.py --validate --seed 42
```

### Phase 2: Baseline Evaluation
```bash
# 1. Run all baseline algorithms
python experiments/run_baselines.py \
    --circuits benchmarks/all_circuits.json \
    --output results/baselines/ \
    --parallel 8 \
    --seed 42

# 2. Statistical analysis
python experiments/analyze_baselines.py \
    --input results/baselines/ \
    --output analysis/baseline_analysis.json
```

### Phase 3: Novel Algorithm Evaluation
```bash
# 1. Property inference algorithm
python experiments/evaluate_property_inference.py \
    --algorithm formalized \
    --circuits benchmarks/all_circuits.json \
    --output results/property_inference/ \
    --runs 5 \
    --seed 42

# 2. Adaptive refinement algorithm
python experiments/evaluate_adaptive_refinement.py \
    --algorithm adaptive \
    --circuits benchmarks/complex_circuits.json \
    --output results/adaptive_refinement/ \
    --runs 5 \
    --seed 42
```

### Phase 4: Comparative Analysis
```bash
# 1. Statistical comparison
python experiments/statistical_comparison.py \
    --baseline results/baselines/ \
    --novel results/property_inference/ \
    --output analysis/comparison.json \
    --alpha 0.05

# 2. Effect size analysis
python experiments/effect_size_analysis.py \
    --results analysis/comparison.json \
    --output analysis/effect_sizes.json
```

### Phase 5: Visualization and Reporting
```bash
# 1. Generate plots
python scripts/generate_plots.py \
    --data analysis/ \
    --output figures/ \
    --format pdf

# 2. Generate report
python scripts/generate_report.py \
    --analysis analysis/ \
    --figures figures/ \
    --output final_report.pdf
```

## Data Collection Protocols

### Automated Data Collection
- **Timestamping**: UTC timestamps for all events
- **Logging**: Structured JSON logs with standardized fields
- **Error Handling**: Graceful failure with detailed error reporting
- **Progress Tracking**: Real-time progress monitoring and ETA estimation

### Manual Data Collection
- **Expert Evaluation**: Blind assessment by verification experts
- **Inter-rater Reliability**: Multiple evaluators with agreement measurement
- **Qualitative Coding**: Systematic categorization of qualitative observations
- **Documentation**: Detailed protocols for manual assessment procedures

### Data Validation
- **Integrity Checks**: Automatic validation of collected data
- **Outlier Detection**: Statistical outlier identification and investigation
- **Consistency Verification**: Cross-validation between automated and manual data
- **Completeness Assessment**: Verification of complete data collection

## Statistical Analysis Framework

### Descriptive Statistics
```python
# Standard descriptive statistics for all metrics
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'time_ms']
for metric in metrics:
    report_statistics(data[metric])  # mean, std, min, max, quartiles
```

### Hypothesis Testing
```python
# Primary hypothesis: Novel algorithm > Baseline
for algorithm_pair in algorithm_comparisons:
    novel_data = results[algorithm_pair['novel']]
    baseline_data = results[algorithm_pair['baseline']]
    
    # Two-tailed t-test
    t_stat, p_value = stats.ttest_ind(novel_data, baseline_data)
    
    # Effect size (Cohen's d)
    effect_size = cohens_d(novel_data, baseline_data)
    
    # Confidence interval
    ci = confidence_interval(novel_data, baseline_data, alpha=0.05)
```

### Power Analysis
```python
# Ensure adequate statistical power (≥ 0.8)
for comparison in planned_comparisons:
    power = calculate_power(
        effect_size=0.5,  # Medium effect size
        alpha=0.05,
        sample_size=comparison['n']
    )
    assert power >= 0.8, f"Insufficient power: {power}"
```

## Quality Assurance

### Code Quality
- **Type Checking**: MyPy static type analysis
- **Linting**: Flake8 style checking
- **Testing**: >90% code coverage with pytest
- **Documentation**: Comprehensive docstring coverage

### Experimental Quality
- **Pilot Studies**: Small-scale validation before full experiments
- **Peer Review**: Internal review of experimental design
- **External Validation**: Independent replication by collaborators
- **Preregistration**: Experimental protocols registered before execution

### Data Quality
- **Validation Rules**: Automated data quality checks
- **Audit Trails**: Complete record of data processing steps
- **Backup Procedures**: Multiple redundant data backups
- **Access Control**: Secure, logged access to experimental data

## Reporting Standards

### Results Reporting
- **Complete Results**: Report all outcomes, including negative results
- **Statistical Details**: Include test statistics, p-values, effect sizes
- **Confidence Intervals**: Report uncertainty in all estimates
- **Practical Significance**: Discuss practical importance beyond statistical significance

### Reproducibility Package
- **Code Archive**: Complete, executable code package
- **Data Package**: Curated datasets with documentation
- **Environment Specification**: Exact software environment description
- **Execution Instructions**: Step-by-step reproduction guide

### Supplementary Materials
- **Extended Results**: Detailed results not fitting in main paper
- **Additional Analyses**: Exploratory analyses and robustness checks
- **Failure Cases**: Analysis of algorithm limitations and failures
- **Implementation Details**: Technical details for complete reproduction

## Timeline and Milestones

### Phase 1: Preparation (Weeks 1-2)
- Environment setup and validation
- Benchmark preparation and verification
- Baseline implementation completion
- Pilot study execution

### Phase 2: Baseline Evaluation (Weeks 3-4)
- Complete baseline algorithm evaluation
- Statistical analysis of baseline performance
- Baseline comparison and validation
- Initial results documentation

### Phase 3: Novel Algorithm Evaluation (Weeks 5-8)
- Property inference algorithm evaluation
- Adaptive refinement algorithm evaluation
- Cross-validation and robustness testing
- Detailed performance analysis

### Phase 4: Comparative Analysis (Weeks 9-10)
- Statistical comparison with baselines
- Effect size and practical significance analysis
- Sensitivity analysis and robustness checks
- Results interpretation and validation

### Phase 5: Documentation and Reporting (Weeks 11-12)
- Figure generation and visualization
- Report writing and review
- Reproducibility package preparation
- Final validation and submission

## Expected Outcomes

### Quantitative Results
- **Performance Improvements**: 2-5x improvement over baselines
- **Statistical Significance**: p < 0.001 for primary comparisons
- **Effect Sizes**: Large effects (d > 0.8) for key metrics
- **Confidence**: High confidence (>0.9) in novel algorithm superiority

### Qualitative Insights
- **Algorithm Strengths**: Detailed characterization of when algorithms excel
- **Failure Modes**: Understanding of algorithm limitations
- **Practical Guidance**: Recommendations for real-world deployment
- **Future Directions**: Identification of promising research directions

### Research Impact
- **Academic Publications**: 4-6 high-impact conference/journal papers
- **Open Source Tools**: Widely adopted verification tools
- **Industry Adoption**: Integration into commercial EDA tools
- **Student Training**: PhD dissertations and research projects

---

This experimental protocol ensures rigorous, reproducible research that advances the field of LLM-assisted hardware verification while maintaining the highest standards of scientific integrity.