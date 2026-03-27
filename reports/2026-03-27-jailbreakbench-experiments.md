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

## Key Results

### Model Family (qwen2.5-7b vs tulu2-7b-sft)

| Model | GCG aurc | Jailbroken aurc | PAIR aurc |
|---|---|---|---|
| qwen2.5-7b-instruct | 2.61 | 7.66 | 7.31 |
| tulu2-7b-sft | 3.62 | 9.07 | 8.10 |

### Pressure Sensitivity (Qwen model sizes, GCG)

| Model | aurc | delta_r |
|---|---|---|
| qwen2.5-0.5b | 4.92 | 0.64 |
| qwen2.5-3b | 2.13 | 0.35 |
| qwen2.5-7b | 2.61 | 0.41 |

For Jailbroken and PAIR attacks, all three Qwen models reach ~94–100% success rate at lambda=10 regardless of size.

### Training Stage (Tulu2-7b variants)

| Model | GCG aurc | Jailbroken aurc |
|---|---|---|
| tulu2-7b-base | 8.68 | 9.44 |
| tulu2-7b-sft | 3.62 | 9.07 |
| tulu2-7b-dpo | 4.67 | 8.09 |

---

## Notable Findings

1. **Attack effectiveness**: Jailbroken and PAIR are much more effective than GCG (aurc 7–9 vs 2–5); GCG shows high variance across models.
2. **Training matters more than size**: DPO and SFT training substantially reduces vulnerability relative to the base model. Larger size alone does not guarantee robustness.
3. **Qwen 3B is surprisingly robust to GCG** (aurc=2.13), outperforming even the 7B model.
4. **Tulu2-7b-base is extremely vulnerable** — near 100% attack success across all attack types.
5. **lambda_star**: Jailbroken/PAIR attacks become effective at low pressure (lambda=1–3); GCG requires higher pressure (lambda=5) or fails entirely.
6. **High-risk harm categories**: Harassment/Discrimination, Malware/Hacking, and Privacy Violations show consistently elevated attack success across models.

---

*Experiments run between March 18–25, 2026. Results stored in `/home/ehghaghi/scratch/ehghaghi/outputs`.*
