# CMAPSS RUL Prediction — MSPatch-iTransformer-RUL

Multi-branch time series Transformer for aircraft engine remaining useful life prediction on the NASA CMAPSS dataset. Part of a 5-person iterative architecture design project.

## Models

| Model | Description | Parameters |
|-------|-------------|------------|
| Base: Patch-iTransformer-RUL | Linear patch + iTransformer encoder | 1.75M / 1.95M |
| Person 1: ConvPatch-iTransformer-RUL | Convolutional patch embedding | 1.75M / 1.95M |
| Person 2: MSPatch-iTransformer-RUL | Multi-scale (3-branch) ConvPatch + fusion | 3.14M / 3.79M |

## Key Results

| Subset | Best Model | RMSE | R² |
|--------|-----------|------|-----|
| FD001 | **Person 2** | **15.27** | **0.86** |
| FD002 | **Person 2** | **27.98** | **0.73** |
| FD003 | **Person 2** | **16.41** | **0.84** |
| FD004 | Person 1 | 25.57 | 0.78 |

Person 2 is the only model achieving R² > 0.72 across all four subsets.

## Project Structure

```
├── main.py                     # Unified experiment entry point
├── src/                        # Source code
│   ├── step1_preprocessing/    # Data loading & normalization
│   ├── step3_models/           # Model definitions (Base, P1, P2)
│   ├── step4_visualization/    # Plotting scripts
│   ├── step5_pipeline/         # Ablation & export scripts
│   └── system/                 # Config & metrics
├── results/                    # Experiment results & figures
├── docs/                       # Reports & documentation
└── skills/                     # Reusable workflow skill
```

## Report

Formal report with embedded figures: [CMAPSS_RUL_技术报告.html](docs/CMAPSS_RUL_技术报告.html)

## Skill

A reusable `transformer-timeseries-workflow` skill is included under `skills/`, capturing the iterative development workflow, common pitfalls, and best practices learned during this project.

## References

- iTransformer (Liu et al., ICLR 2024)
- PatchTST (Nie et al., ICLR 2023)
- TTSNet (Li et al., Sensors 2025)
- NASA C-MAPSS dataset (Saxena et al., PHM08)
