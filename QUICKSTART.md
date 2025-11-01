# Quick Start Guide

Get started with your reproducibility study in 5 steps.

## Step 1: Identify and Understand the Original Paper

- [ ] Select the paper to reproduce
- [ ] Read the paper thoroughly
- [ ] Identify key claims to reproduce
- [ ] Check for code/data availability
- [ ] Document your initial assessment

**Action:** Fill in Section 2 (Original Paper Summary) of `paper.md`

## Step 2: Set Up Your Environment

- [ ] Clone/download original code (if available)
- [ ] Set up computational environment
- [ ] Install dependencies
- [ ] Obtain datasets
- [ ] Verify original code runs

**Action:** 
1. Add original code to `code/original/`
2. Document setup in `code/README.md`
3. Place data in `data/raw/` (or document access)

## Step 3: Plan Your Reproduction

- [ ] Define scope of reproduction
- [ ] Identify required resources
- [ ] Set timeline and milestones
- [ ] Plan experiments to run
- [ ] Start experimental log

**Action:**
1. Fill in Section 3 (Reproducibility Methodology) of `paper.md`
2. Start logging in `EXPERIMENTAL_LOG.md`
3. Begin `REPRODUCIBILITY_CHECKLIST.md`

## Step 4: Execute Reproduction

- [ ] Implement/adapt code as needed
- [ ] Run experiments with multiple seeds
- [ ] Record all results systematically
- [ ] Document challenges and solutions
- [ ] Create visualizations

**Action:**
1. Work in `code/reproduced/` or `code/scripts/`
2. Save results to `results/raw/`
3. Create figures in `figures/reproduced/`
4. Keep `EXPERIMENTAL_LOG.md` updated

## Step 5: Analyze and Document

- [ ] Compare results with original
- [ ] Perform statistical analysis
- [ ] Complete reproducibility assessment
- [ ] Write up findings
- [ ] Complete all templates

**Action:**
1. Fill in `COMPARISON_TEMPLATE.md`
2. Complete Sections 5-9 of `paper.md`
3. Finalize `REPRODUCIBILITY_CHECKLIST.md`
4. Review and revise entire document

---

## Daily Workflow

### Morning
1. Review yesterday's progress
2. Set today's goals
3. Update experimental log

### During Work
1. Make changes incrementally
2. Document as you go
3. Run experiments systematically
4. Save all outputs

### Evening
1. Summarize the day in experimental log
2. Commit code changes
3. Back up results
4. Plan tomorrow's work

---

## File-by-File Guide

### Start Here
1. **README.md** - Overview and instructions
2. **QUICKSTART.md** - This file
3. **STRUCTURE.md** - Understand paper structure

### Main Paper
4. **paper.md** - The actual paper (fill this in progressively)

### Tracking Progress
5. **EXPERIMENTAL_LOG.md** - Daily log of work
6. **REPRODUCIBILITY_CHECKLIST.md** - Track reproducibility aspects

### Analysis Templates
7. **COMPARISON_TEMPLATE.md** - Compare your results to original

### Supporting Directories
8. **code/** - All code
9. **data/** - All datasets
10. **results/** - All experimental outputs
11. **figures/** - All visualizations

---

## Common Workflows

### Running an Experiment

```bash
# 1. Navigate to code directory
cd code

# 2. Run experiment
python reproduced/main.py --config configs/experiment1.yaml

# 3. Move results to results directory
mv outputs/* ../results/raw/experiment1_$(date +%Y%m%d)/

# 4. Analyze results
python scripts/analyze_results.py

# 5. Generate figures
python scripts/plot_results.py

# 6. Update experimental log
# (Document what you learned)
```

### Comparing Results

```bash
# 1. Extract metrics from original paper
# (Manual or automated extraction)

# 2. Extract metrics from your results
python scripts/extract_metrics.py

# 3. Create comparison
python scripts/compare_results.py

# 4. Update COMPARISON_TEMPLATE.md with findings
```

### Creating Figures

```bash
# 1. Generate reproduced figure
python scripts/plot_figure1.py --data results/processed/exp1.csv

# 2. Place in figures/reproduced/
mv figure1.png figures/reproduced/

# 3. Create comparison if possible
python scripts/compare_figures.py

# 4. Update figures/README.md
```

---

## Tips for Success

### Organization
- Keep directories clean and organized
- Use consistent naming conventions
- Document everything immediately
- Commit frequently

### Experimentation
- Run multiple seeds (at least 3-5)
- Save all hyperparameters
- Keep raw outputs before processing
- Record resource usage

### Documentation
- Write as you go, not at the end
- Be honest about failures
- Provide sufficient detail
- Include examples

### Time Management
- Set realistic milestones
- Don't get stuck on one issue too long
- Ask for help when needed
- Take breaks to avoid burnout

---

## Troubleshooting

### "I can't get the original code to run"
- Document the exact errors
- Try different environment setups
- Search for known issues
- Consider implementing from scratch
- Document attempts in experimental log

### "My results don't match"
- Run multiple times with different seeds
- Check for undocumented hyperparameters
- Verify data preprocessing
- Look for version differences in dependencies
- Document differences objectively

### "I'm spending too much time"
- Reassess scope - maybe partial reproduction
- Document time in experimental log
- This information is valuable for the paper
- Consider what's achievable in your timeframe

### "The original paper lacks details"
- Document what's missing
- Make reasonable assumptions
- Explain your choices
- This is important information for the paper

---

## Checklist for Completion

### Before Submission
- [ ] All experiments completed
- [ ] Results documented and compared
- [ ] All figures created
- [ ] Paper sections filled in
- [ ] Reproducibility checklist completed
- [ ] Code is well-documented
- [ ] Code is publicly available (if possible)
- [ ] Data access documented
- [ ] All claims are substantiated
- [ ] Writing is clear and objective
- [ ] Paper has been reviewed by others
- [ ] Supplementary materials prepared

---

## Need Help?

- Review `STRUCTURE.md` for section guidance
- Check README files in each directory
- Look at templates for examples
- Consult reproducibility guidelines from major venues

## Next Steps

1. Choose your paper to reproduce
2. Read through `STRUCTURE.md`
3. Start filling in Section 2 of `paper.md`
4. Begin your experimental log
5. Work through Steps 1-5 above

Good luck with your reproducibility study!
