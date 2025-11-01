# Figures Directory

This directory contains all figures, plots, and visualizations for your reproducibility paper.

## Organization

### original/
Figures from the original paper
- Screenshots or extracted figures
- For reference and comparison

### reproduced/
Your reproduced figures
- Should match original figures as closely as possible
- Use consistent styling and formatting

### comparison/
Side-by-side comparisons
- Original vs. reproduced visualizations
- Difference plots
- Correlation plots

### supplementary/
Additional figures for appendices
- Extended results
- Ablation studies
- Error analyses

## Example Structure

```
figures/
├── README.md
├── original/
│   ├── fig1_original.png
│   └── fig2_original.png
├── reproduced/
│   ├── fig1_reproduced.png
│   └── fig2_reproduced.png
├── comparison/
│   ├── fig1_comparison.png
│   └── metrics_comparison.png
└── supplementary/
    ├── error_analysis.png
    └── ablation_study.png
```

## Figure Guidelines

### Quality Standards
- **Resolution:** Minimum 300 DPI for publications
- **Format:** PNG for screenshots, PDF/SVG for plots
- **Size:** Appropriate for publication (typically 3-7 inches width)
- **Colors:** Colorblind-friendly palettes
- **Text:** Readable font sizes (minimum 8pt)

### Naming Convention

Use descriptive names: `[figure_number]_[description]_[version].ext`

Examples:
- `fig1_accuracy_comparison_v1.png`
- `fig2_loss_curves_reproduced.pdf`
- `figS1_ablation_study.png` (supplementary)

### Caption Template

For each figure, document:

```markdown
**Figure [X]:** [Title]

[Detailed caption explaining what the figure shows]

- **Panel A:** [Description]
- **Panel B:** [Description]

**Comparison with Original:** [How it compares]
```

## Figure Inventory

### Main Figures

| Figure # | Title | Original | Reproduced | Status |
|----------|-------|----------|------------|--------|
| 1 | | ✓ | ✓ | Match/Differ/Missing |
| 2 | | ✓ | ✓ | Match/Differ/Missing |

### Supplementary Figures

| Figure # | Title | Description | Status |
|----------|-------|-------------|--------|
| S1 | | | Complete/Pending |

## Reproduction Notes

### Figure 1
- **Original Source:** [Paper page/section]
- **Data Source:** [Where data comes from]
- **Reproduction Method:** [How you recreated it]
- **Differences:** [Any notable differences]
- **Match Quality:** [Excellent/Good/Fair/Poor]

### Figure 2
- **Original Source:** 
- **Data Source:** 
- **Reproduction Method:** 
- **Differences:** 
- **Match Quality:** 

## Visualization Code

Document the code used to generate each figure:

```python
# Example: Figure 1
# Script: code/scripts/plot_figure1.py
# Dependencies: matplotlib, seaborn, numpy
# Data: results/processed/experiment_1_summary.csv
```

## LaTeX Integration

Example LaTeX code for including figures:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\columnwidth]{figures/reproduced/fig1_reproduced.pdf}
    \caption{[Caption text]}
    \label{fig:reproduced_1}
\end{figure}
```

## Checklist

For each main figure:
- [ ] Original figure obtained/extracted
- [ ] Reproduced version created
- [ ] Comparison created (if applicable)
- [ ] Caption written
- [ ] Differences documented
- [ ] Quality verified
- [ ] Files named correctly
- [ ] Included in paper

## Notes

[Any additional notes about figures and visualizations]
