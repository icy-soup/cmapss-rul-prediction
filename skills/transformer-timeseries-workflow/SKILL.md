---
name: transformer-timeseries-workflow
description: >-
  Complete iterative development workflow for Transformer-based time series forecasting and regression projects.
  Covers data preprocessing verification, model architecture patterns (Patch + iTransformer + multi-scale),
  baseline establishment, ablation studies, result analysis, and report generation.
  Documents common pitfalls discovered during real project experience.
  Use this skill whenever starting a time series Transformer project, designing experiments,
  debugging model convergence issues, planning ablation studies, or setting up a multi-model comparison pipeline.
  Especially relevant for projects involving sensor data, RUL prediction, or multi-condition time series.
type: rigid
---

# Transformer Time Series Iterative Development Workflow

A structured workflow for developing Transformer-based time series prediction/regression models. Designed for projects that involve multiple model variants, iterative architecture improvements, and systematic comparison.

**Core principle:** Verify data preprocessing correctness BEFORE optimizing model architecture. Many "architecture limitations" turn out to be data issues.

---

## Phase 1: Problem Definition & Literature Review

### 1.1 Understand the Dataset

Before writing any code, understand the data structure:

- How many samples, features, time steps?
- Are there multiple conditions/regimes in the data (e.g., different operating conditions)?
- Is the task forecasting (predict future values) or regression (predict a target from a window)?
- What is the evaluation metric? Is it symmetric (RMSE, MAE) or asymmetric (custom score)?

### 1.2 Survey Relevant Works

Identify key architecture patterns from recent literature:

| Pattern | Key Reference | When to Use |
|---------|--------------|-------------|
| Time Series Patch (divide sequence into patches) | PatchTST (Nie 2023) | Default: almost always beneficial |
| Inverted Transformer (variables as tokens) | iTransformer (Liu 2024) | Multi-variate, cross-variable dependencies matter |
| Convolutional Patch Embedding | ConvPatch variants | Local pattern extraction within patches |
| Hierarchical / Multi-scale | TTSNet, Hierarchical Transformer | Data has patterns at multiple timescales |
| Sparse / Grouped Attention | Longformer, GroupSparse | Many variables, reducing noise |

### 1.3 Define Baseline Strategy

Plan a clear baseline-to-improvement path:
1. **Baseline:** Simplest reasonable architecture (e.g., linear patch + Transformer)
2. **Iteration 1:** Single targeted improvement (e.g., conv patch instead of linear)
3. **Iteration 2:** Next improvement on top (e.g., multi-scale branches)
4. **Iteration N:** Continue until marginal gain diminishes

---

## Phase 2: Data Preprocessing

This is where most bugs hide. A preprocessing error can look like an architecture limitation.

### 2.1 Critical Checklist

- [ ] **Normalization:** Is it global or per-group? If the data has multiple operating conditions, per-group normalization may be necessary. Test by visualizing sensor distributions per condition.
- [ ] **Metadata features:** Are there known condition variables (e.g., altitude, speed, temperature) that are being discarded? These can be critical additional input channels for multi-condition datasets.
- [ ] **Train/test consistency:** Test data must be normalized using TRAINING set statistics, not its own.
- [ ] **Window/patch parameters:** Does the window size cover relevant temporal patterns? Does patch stride create overlap?
- [ ] **Target encoding:** If using capped/clipped targets (e.g., RUL capped at 125), verify this is standard practice in the field by checking relevant papers.

### 2.2 Common Pitfalls

**Pitfall 1: Normalization hides condition information.**
When data comes from multiple distinct operating conditions, global normalization mixes their distributions together. The model cannot distinguish which condition a sample belongs to, leading to degraded performance that looks like an "architecture limitation."

*Fix:* Cluster by operating condition parameters and normalize per cluster. Also feed condition parameters as additional input features where appropriate.

**Pitfall 2: Metadata silently discarded.**
During data loading, condition variables (altitude, speed, etc.) are loaded but never passed to the model. The model only sees sensor readings without context. Performance suffers, and the blame goes to the model architecture.

*Fix:* Check the actual input tensor shape. Add condition variables as additional channels for multi-condition subsets.

**Pitfall 3: Test set normalized independently.**
Test data is normalized with its own mean/std instead of the training set's. This causes train/test distribution shift that systematically hurts evaluation metrics.

*Fix:* Compute normalization statistics on training data only; apply the same statistics to test data.

---

## Phase 3: Baseline Model

### 3.1 Architecture Template

Start with a minimal architecture and verify it works before adding complexity:

```
Input (B, L, C)
  → Transpose to (B, C, L)                    # variables as channels
  → Patch Embedding (Linear or Conv)          # (B, C, N, d_model)
  → Flatten patches: (B, C, N*d_model)        # per-variable feature
  → iTransformer Encoder                      # cross-variable attention
  → Pool over variables (mean)                # (B, d_model)
  → MLP Regressor                             # (B,)
```

### 3.2 Hyperparameter Defaults

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| d_model | 128-256 | Embedding dimension |
| n_heads | 4-8 | Attention heads |
| e_layers | 2-4 | Transformer encoder layers |
| d_ff | 256-1024 | Feed-forward hidden dim |
| dropout | 0.2-0.3 | Regularization |
| learning rate | 1e-4 - 5e-4 | AdamW optimizer |
| batch_size | 64-128 | Depends on data size |
| warmup_epochs | 10 | Linear warmup |
| patience | 30-50 | Early stopping |

### 3.3 Verify Baseline Before Iterating

- [ ] Training loss decreases consistently (log-scale plot helps)
- [ ] Validation loss does not diverge for extended periods
- [ ] Predictions have reasonable dynamic range (not collapsed to mean)
- [ ] R² > 0 on held-out data (model learns something)
- [ ] Run at least 2 seeds to verify stability

---

## Phase 4: Iterative Architecture Improvement

### 4.1 Make One Change at a Time

Each iteration should introduce exactly one new architectural element. This makes ablation studies meaningful.

**Common iteration patterns:**

1. **Linear Patch → Conv Patch:** Replace linear projection with depthwise+pointwise convolution (kernel=5). Benefit: better local pattern extraction.
2. **Single-scale → Multi-scale:** Add parallel branches with different patch sizes (e.g., P=8/S=4, P=16/S=8, P=32/S=16). Benefit: capture patterns at multiple temporal resolutions.
3. **Concat Fusion → Gated Fusion:** Experiment with different fusion mechanisms for multi-branch architectures.
4. **Standard Transformer → iTransformer:** Swap time-as-token for variable-as-token attention.

### 4.2 Fusion Strategies (for Multi-branch Architectures)

| Method | How It Works | Best For |
|--------|-------------|----------|
| Concat + Linear | Concatenate branch outputs, project to d_model | Simple, effective default |
| Weighted Sum | Project each branch to d_model, weighted average | Lightweight, few params |
| Gated Fusion | Per-variable gating via sigmoid network | Adaptive feature selection |

Concat is the simplest and often the best. More complex fusion adds parameters without guaranteed improvement.

### 4.3 Track Parameters

Log parameter counts for every model variant. A meaningful improvement should justify its parameter cost:

```
Baseline:   1.75M params, RMSE=19.8
Variant A:  1.75M params, RMSE=18.3  (+0% params, -8% RMSE)
Variant B:  3.14M params, RMSE=15.3  (+80% params, -23% RMSE)
```

---

## Phase 5: Ablation & Validation

### 5.1 Ablation Study Design

Test each architectural choice independently:

**Branch count ablation (Leave-One-Branch-Out):** Compare N-branch configurations on a representative subset. Include all single branches, all pairwise combinations, and the full model.
```
Results example:
  single-small (P=8/S=4):     RMSE 15.9, 2.21M params
  single-medium (P=16/S=8):   RMSE 16.1, 1.95M params
  single-large (P=32/S=16):   RMSE 15.7, 1.82M params
  small+medium:               RMSE 13.9, 2.60M params  ← best
  3-branch all:               RMSE 14.6, 3.14M params
```

**Fusion ablation:** Compare fusion methods on the same multi-branch architecture.

### 5.2 Control Experiments

These experiments determine whether multi-branch improvement comes from genuine multi-scale diversity or simply from having more parameters.

**Pseudo multi-branch control:** Create multiple branches with the SAME patch size (e.g., 3 branches all using P16S8). If performance drops compared to diverse scales, it confirms that multi-scale diversity drives the improvement, not just having parallel branches.

**Capacity-matched single branch:** Increase the single branch's capacity (double d_model, add layers) to match the multi-branch parameter count. If the multi-branch still outperforms, it demonstrates that multi-scale information—not raw capacity—is responsible.

### 5.3 Representation Analysis (Training-Free)

Analyze what each branch actually learns using existing trained models:

**CKA similarity:** Compute Centered Kernel Alignment between branch representations. High similarity (>0.9) suggests redundancy; moderate similarity (0.6-0.85) suggests genuine complementarity.

**Per-target bucket analysis:** Group test samples by target range (e.g., early/mid/late life stages). Check whether different branches excel at different stages. This directly tests whether multi-scale design achieves its intended effect.

**Gradient analysis:** Compare gradient magnitudes across branches to identify which branches contribute most to learning at different training phases.

### 5.4 Validation Across Conditions

If the data has multiple conditions/subgroups, validate on each separately. A model that works well on simple data may fail on complex data, and vice versa.

### 5.5 Multiple Seeds & Statistical Testing

Single-run results can be misleading due to random seed variation (train/val split, initialization). Critical configurations should be verified with at least 3 seeds.

**Interpreting multi-seed results:**
- If mean difference > std dev: likely real improvement
- If mean difference < std dev: within noise, treat as equivalent
- If one config has much lower variance: it is more robust, even if mean is slightly worse
- For formal comparisons, use a paired t-test across matched seeds to determine statistical significance

---

## Phase 6: Results Analysis & Visualization

### 6.1 Essential Visualizations

1. **Training curves** — loss (log scale) + validation RMSE per epoch
2. **Prediction scatter** — predicted vs true with y=x reference line, residual histogram
3. **Bar chart comparison** — RMSE + Score across all models and subsets
4. **Per-sample/engine predictions** — sorted by true value, show prediction tracking

### 6.2 Coordinate Axis Check

Always verify plot axis limits are reasonable:
- Scatter plots: if predictions are capped (e.g., RUL ≤ 125) but true values go higher, cap the axis to match the model's output range. Otherwise, empty space in the plot misleads the viewer.
- Comparison charts: add padding so value labels don't touch borders or overlap with adjacent bars.

### 6.3 Metrics Table

Present results in a consistent format across all model variants:

| Model | Subset | RMSE | R² | Score | Params |
|-------|--------|------|-----|-------|--------|
| Baseline | A | 19.8 | 0.77 | 1085 | 1.75M |
| Variant X | A | **15.3** | **0.86** | **439** | 3.14M |

### 6.4 Check for Collapse

If a model's predictions are nearly constant (prediction range < 10), it has collapsed:
- First check data preprocessing — this is almost always the root cause
- Verify normalization is appropriate for the data structure
- Verify the model receives all necessary input features
- Only after eliminating data issues, consider architecture changes

---

## Phase 7: Report & Documentation

### 7.1 Report Structure

A complete report should include:

1. **Architecture overview** — text description + flow diagram
2. **Experimental setup** — hyperparameters, data processing, evaluation metrics
3. **Main results** — comparison table across all models and subsets
4. **Visualizations** — training curves, scatter plots, comparison charts
5. **Ablation studies** — branch count, fusion method, per-condition validation
6. **Discussion** — what worked, what didn't, why
7. **Conclusion** — key findings, limitations, future directions

### 7.2 Data Preprocessing Documentation

Document every preprocessing choice with justification:
- Sensor/feature selection
- Normalization strategy (global vs per-condition)
- Target encoding (capped or uncapped, rationale with citations)
- Train/validation split strategy

### 7.3 Visual Quality Checklist

- [ ] Figure titles are descriptive
- [ ] Axis labels are readable
- [ ] Labels don't overlap with data points or other text
- [ ] Color coding is consistent across figures
- [ ] Font sizes are appropriate for the figure dimensions
- [ ] Legends are placed where they don't obscure data

---

## Consolidated Common Pitfalls

1. **Normalization bug misdiagnosed as architecture limitation.** Multi-condition data needs per-condition normalization. Symptoms: model collapses on complex subsets, performs OK on simple subsets. Fix preprocessing first, not architecture.

2. **Metadata features silently dropped.** Condition variables loaded but never fed to the model. The model lacks context to distinguish different regimes. Check input tensor shapes.

3. **Test set normalized with its own statistics.** This creates train/test distribution shift. Always normalize test data with training set statistics.

4. **Single-run variance mistaken for significant improvement.** A 0.5 RMSE difference across seeds can be noise. Run multiple seeds for critical comparisons.

5. **Axis limits waste plot space.** If predictions are capped but true values extend far beyond, the plot has empty space. Cap axes to the relevant range.

6. **Target capping is dataset-specific.** Check relevant papers before deciding whether to cap regression targets. Some benchmarks have established conventions.

7. **Assuming architecture is the bottleneck too early.** Data preprocessing quality often matters more than model architecture, especially for multi-condition datasets. Verify data first.
