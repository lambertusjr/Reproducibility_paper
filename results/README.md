# Results Directory

This directory contains all experimental results from your reproducibility study.

## Organization

### raw/
Raw output from experiments
- Unprocessed experiment outputs
- Logs and traces
- Model checkpoints

### processed/
Processed and analyzed results
- Aggregated metrics
- Statistical analyses
- Comparison tables

### comparisons/
Side-by-side comparisons with original paper
- Tables comparing metrics
- Difference analyses
- Statistical test results

## Example Structure

```
results/
├── README.md
├── raw/
│   ├── experiment_1/
│   │   ├── run_1/
│   │   ├── run_2/
│   │   └── run_3/
│   ├── experiment_2/
│   └── logs/
├── processed/
│   ├── experiment_1_summary.csv
│   ├── experiment_2_summary.csv
│   └── aggregate_metrics.json
└── comparisons/
    ├── experiment_1_comparison.csv
    ├── experiment_2_comparison.csv
    └── statistical_tests.txt
```

## Results Format

### Experiment Naming Convention

Use consistent naming: `[experiment_name]_[date]_[run_number]`

Example: `baseline_20240101_run1`

### Required Information

For each experiment, document:
- Date and time
- Configuration/hyperparameters used
- Random seed
- Hardware used
- Software versions
- Runtime
- Resource usage (CPU, GPU, memory)

### Metadata Template

```json
{
  "experiment_name": "",
  "date": "YYYY-MM-DD",
  "run_number": 1,
  "config": {},
  "seed": 42,
  "hardware": "",
  "runtime_seconds": 0,
  "metrics": {}
}
```

## Results Summary

### Experiment 1: [Name]

| Run | Metric 1 | Metric 2 | Metric 3 | Runtime |
|-----|----------|----------|----------|---------|
| 1   |          |          |          |         |
| 2   |          |          |          |         |
| 3   |          |          |          |         |
| Mean|          |          |          |         |
| Std |          |          |          |         |

### Experiment 2: [Name]

| Run | Metric 1 | Metric 2 | Metric 3 | Runtime |
|-----|----------|----------|----------|---------|
| 1   |          |          |          |         |
| 2   |          |          |          |         |
| 3   |          |          |          |         |
| Mean|          |          |          |         |
| Std |          |          |          |         |

## Comparison with Original

### Experiment 1

| Metric | Original | Reproduced (Mean ± Std) | Difference |
|--------|----------|-------------------------|------------|
|        |          |                         |            |

### Experiment 2

| Metric | Original | Reproduced (Mean ± Std) | Difference |
|--------|----------|-------------------------|------------|
|        |          |                         |            |

## Statistical Analysis

[Document statistical tests performed]

### Test 1: [Test Name]

- **Null Hypothesis:** [H0]
- **Result:** [Accept/Reject]
- **P-value:** [value]
- **Interpretation:** [What this means]

## Computational Resources

### Total Resources Used

| Resource | Amount | Cost (if applicable) |
|----------|--------|---------------------|
| CPU Hours|        |                     |
| GPU Hours|        |                     |
| Memory   |        |                     |
| Storage  |        |                     |

### Per-Experiment Resources

| Experiment | CPU Time | GPU Time | Memory | Storage |
|------------|----------|----------|--------|---------|
|            |          |          |        |         |

## Notes

[Any additional notes about results]
