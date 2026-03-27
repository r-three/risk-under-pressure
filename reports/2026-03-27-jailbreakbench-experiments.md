# JailbreakBench Experiment Summary
**Date:** March 27, 2026

## Experiment Groups

| Group | Models | n_prompts |
|---|---|---|
| **Model Family** | qwen2.5-7b-instruct, tulu2-7b-sft | 100 |
| **Pressure Sensitivity** | qwen2.5-{0.5b,3b,7b}-instruct | 50 & 100 |
| **Training Stage** | tulu2-7b-{base,sft,dpo} | 50 & 100 |

## Attacks Evaluated

- **GCG** — gradient-based optimization
- **PAIR** — iterative prompt refinement
- **Jailbroken** — adversarial prompt templates

## Metrics

- `aurc` — Area Under Risk Curve (aggregate vulnerability)
- `delta_r` — max risk at lambda=10
- `lambda_star` — pressure threshold where attack becomes effective
- Risk curves at lambda ∈ {0, 1, 3, 5, 10}

## Harm Categories

10 categories were evaluated: Disinformation, Economic Harm, Expert Advice, Fraud/Deception, Government Decision-Making, Harassment/Discrimination, Malware/Hacking, Physical Harm, Privacy Violations, Sexual/Adult Content.

---

## Model Family

Comparing `qwen2.5-7b-instruct` vs `tulu2-7b-sft` across all three attacks.

| Model | GCG aurc | Jailbroken aurc | PAIR aurc |
|---|---|---|---|
| qwen2.5-7b-instruct | 2.61 | 7.66 | 7.31 |
| tulu2-7b-sft | 3.62 | 9.07 | 8.10 |

![Risk-Pressure Curves — Model Family](/home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots/risk_curves_combined.png)

Tulu2-7b-sft is more vulnerable across all attacks, especially Jailbroken and PAIR which reach near 100% at lambda=10. GCG is notably weaker for both models.

### Category Exploitability — Break Pressure

![Break Pressure by Category — Model Family](/home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots/break_pressure_by_category.png)

Sexual/Adult content and Physical harm are the easiest categories to exploit (lowest break pressure), while Expert Advice and Privacy are the most resistant.

### Category Heatmaps at λ=10

| GCG | Jailbroken | PAIR |
|---|---|---|
| ![](/home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots/heatmap_category_gcg.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots/heatmap_category_jailbroken.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots/heatmap_category_pair.png) |

---

## Pressure Sensitivity

How risk scales with lambda across three Qwen model sizes.

| Model | GCG aurc | GCG delta_r |
|---|---|---|
| qwen2.5-0.5b | 4.92 | 0.64 |
| qwen2.5-3b | 2.13 | 0.35 |
| qwen2.5-7b | 2.61 | 0.41 |

For Jailbroken and PAIR attacks, all three models reach ~94–100% success rate at lambda=10 regardless of size.

![Risk-Pressure Curves — Pressure Sensitivity](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/risk_curves_combined.png)

The 3B model is surprisingly the most robust to GCG. For prompt-based attacks (Jailbroken, PAIR), model size provides no meaningful protection.

### Per-Attack Risk Curves

| GCG | Jailbroken | PAIR |
|---|---|---|
| ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/risk_curves_gcg.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/risk_curves_jailbroken.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/risk_curves_pair.png) |

### Category Heatmaps at λ=10

| GCG | Jailbroken | PAIR |
|---|---|---|
| ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/heatmap_category_gcg.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/heatmap_category_jailbroken.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots/heatmap_category_pair.png) |

---

## Training Stage

How the training recipe (base → SFT → DPO) affects robustness on Tulu2-7b.

| Model | GCG aurc | Jailbroken aurc |
|---|---|---|
| tulu2-7b-base | 8.68 | 9.44 |
| tulu2-7b-sft | 3.62 | 9.07 |
| tulu2-7b-dpo | 4.67 | 8.09 |

![Risk-Pressure Curves — Training Stage](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/risk_curves_combined.png)

The base model is extremely vulnerable to all attacks. SFT provides a large improvement for GCG. DPO further reduces vulnerability to Jailbroken/PAIR relative to SFT.

### Per-Attack Risk Curves

| GCG | Jailbroken | PAIR |
|---|---|---|
| ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/risk_curves_gcg.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/risk_curves_jailbroken.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/risk_curves_pair.png) |

### Category Heatmaps at λ=10

| GCG | Jailbroken | PAIR |
|---|---|---|
| ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/heatmap_category_gcg.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/heatmap_category_jailbroken.png) | ![](/home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots/heatmap_category_pair.png) |

---

## Notable Findings

1. **Attack effectiveness**: Jailbroken and PAIR are much more effective than GCG (aurc 7–9 vs 2–5); GCG shows high variance across models.
2. **Training matters more than size**: DPO and SFT substantially reduce vulnerability relative to base. Larger size alone does not guarantee robustness.
3. **Qwen 3B is surprisingly robust to GCG** (aurc=2.13), outperforming the 7B model.
4. **Tulu2-7b-base is extremely vulnerable** — near 100% attack success across all attack types.
5. **lambda_star**: Jailbroken/PAIR attacks become effective at low pressure (lambda=1–3); GCG requires higher pressure (lambda=5) or fails entirely.
6. **High-risk harm categories**: Harassment/Discrimination, Malware/Hacking, and Privacy Violations show consistently elevated attack success across models.

---

*Experiments run between March 18–25, 2026. Results stored in `/home/ehghaghi/scratch/ehghaghi/outputs`.*
