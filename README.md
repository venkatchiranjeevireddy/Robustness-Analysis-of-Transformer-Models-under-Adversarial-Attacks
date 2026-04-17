# Transformer Robustness Evaluation under Adversarial Attacks

> Systematic analysis of how DistilBERT, BERT, and RoBERTa hold up under real-world adversarial conditions — with defense mechanisms, failure diagnostics, and structured insights.

---

## Overview

Modern transformer models achieve strong benchmark performance, but that performance can collapse under even minor perturbations to input text. This project evaluates the **robustness of three transformer architectures** — DistilBERT, BERT, and RoBERTa — against a suite of adversarial text attacks commonly encountered in production environments.

The goal is not just to measure accuracy drops, but to **understand why models fail**: which attack types are most damaging, what tokenization behaviors create vulnerabilities, and which defense strategies actually hold up.

This project is built for engineers and researchers who care about **reliability, not just benchmark scores**.

---

## Key Features

- **Multi-model evaluation** — DistilBERT (baseline + adversarially trained variants), RoBERTa
- **6 adversarial attack types** — covering noise, semantics, structure, and negation
- **3 defense strategies** — adversarial training (basic + V2), input normalization/preprocessing
- **Structured failure analysis** — categorized failure modes with per-case explanations
- **Robustness scoring** — beyond accuracy; dedicated metric for cross-condition stability
- **Insight generation** — automated analysis of *why* each model/attack behaves as it does
- **Interview-ready report** — final structured summary with rankings, trade-offs, and takeaways

---

## System Pipeline

```
Raw Text Input
      │
      ▼
┌─────────────────────┐
│  Adversarial Attack  │  ◀── Synonym / Typo / Dropout / Negation / POS Synonym / Phrase
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Preprocessing /     │  ◀── Normalization, spell correction, Unicode cleanup
│  Defense Layer       │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Transformer Model   │  ◀── DistilBERT / RoBERTa (clean or adversarially trained)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Evaluation Engine   │  ◀── Accuracy · F1 Score · Robustness Score
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Analysis Layer      │  ◀── Failure categorization · Case analysis · Insights
└─────────────────────┘
      │
      ▼
  Final Report
```

---

## Models Used

| Model | Description |
|---|---|
| **DistilBERT Baseline** | Lightweight BERT variant; fast inference, standard fine-tuning |
| **DistilBERT + Norm** | Baseline + input normalization preprocessing |
| **DistilBERT Adv Basic** | Adversarially trained on a subset of attack types |
| **DistilBERT Adv V2** | Adversarially trained on all 6 attack conditions |
| **RoBERTa Baseline** | Robustly optimized BERT with larger pre-training corpus |

---

## Adversarial Attacks

| Attack | Description |
|---|---|
| **Synonym Replacement** | Replaces content words with synonyms. Targets models over-reliant on specific high-frequency tokens. |
| **Typo Injection** | Introduces character-level noise (misspellings). Breaks sub-word tokenization, producing unknown or incorrect token pieces. |
| **Token Dropout** | Randomly removes tokens from the input. Destroys positional context and forces prediction from incomplete evidence. |
| **Negation Insertion** | Inserts or flips negation words (`not`, `never`). Inverts sentence polarity — highly effective against attention-based models that underweight negation tokens. |
| **POS Synonym** | Substitutes words with part-of-speech-aware synonyms. Keeps grammar intact while shifting meaning — harder to detect than noise-based attacks. |
| **Phrase Perturbation** | Injects short distractor phrases. Mildly disrupts the global `[CLS]` representation; least damaging of the six attacks. |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Standard classification accuracy on attacked vs. clean inputs |
| **F1 Score** | Harmonic mean of precision and recall; handles class imbalance |
| **Robustness Score** | Average accuracy maintained across all adversarial conditions, normalized against clean baseline. A model with high clean accuracy but steep drops under attack will score lower here. |

---

## Key Results

### Model Robustness Ranking

| Rank | Model | Robustness Score |
|---|---|---|
| 🥇 1 | DistilBERT Adv Basic | 0.9858 |
| 🥈 2 | RoBERTa Baseline | 0.9818 |
| 🥉 3 | DistilBERT Baseline | 0.9706 |
| 4 | DistilBERT + Norm | 0.9701 |
| 5 | DistilBERT Adv V2 | 0.9548 |

### Attack Damage on DistilBERT Baseline

| Attack | Accuracy Drop |
|---|---|
| Dropout | -5.33% |
| Synonym | -3.00% |
| Typo | -2.00% |
| Negation | -2.00% |
| POS Synonym | -2.00% |
| Phrase | -0.67% |

### Notable Findings

- **Token Dropout is the hardest attack** — random removal destroys context that attention cannot reconstruct
- **Adversarial training (basic) outperforms V2** — training on all 6 attacks simultaneously dilutes the benefit per attack; focused training generalizes better
- **Larger pre-training ≠ more robustness** — RoBERTa has higher clean accuracy (0.8867) but loses to DistilBERT Adv Basic on robustness score
- **Preprocessing alone is insufficient** — normalization helps only against surface noise; semantic attacks (negation, synonym) pass through unchanged

---

## Failure Analysis

### Failure Category Distribution

| Category | Weight | Share |
|---|---|---|
| Noise / Typo Sensitivity | 7.33 | ~54% |
| Synonym / Context Confusion | 5.00 | ~37% |
| Negation / Polarity Failure | 2.00 | ~15% |
| Structural / Phrase Distraction | 0.67 | ~5% |

### Root Cause Patterns

**1. Tokenization Sensitivity**
Character-level perturbations (typos) cause the WordPiece / BPE tokenizer to split words into incorrect or unknown sub-word pieces. When the primary sentiment-bearing token becomes `[UNK]` or a meaningless fragment, the model loses its strongest signal entirely. Example: `"Terrible"` → `"Terribel"` → tokenized as unrecognized pieces → NEGATIVE prediction flips to POSITIVE.

**2. Negation Blindness**
Transformers attend to high-salience content words. Negation tokens (`not`, `never`) receive relatively low attention weight, especially when surrounded by strong positive vocabulary. The model reads the dominant positive words and ignores the polarity inversion. Example: `"absolutely wonderful"` still triggers POSITIVE even after inserting `"not"`.

**3. Context Loss from Dropout**
When key tokens are removed, surviving fragments lose grammatical and semantic coherence. The model is forced to predict from partial evidence — shorter, broken sequences push confidence toward the wrong class. Example: removing `"enjoyed"`, `"film"`, `"acting"` from a positive review results in a NEGATIVE prediction.

**4. Synonym Boundary Shift**
Models learn decision boundaries around high-frequency sentiment words seen during fine-tuning. Low-frequency synonyms (`"futile"` for `"useless"`, `"squander"` for `"waste"`) exist closer to the neutral region of the embedding space, nudging predictions off the correct class.

---

## Defense Mechanisms

### Adversarial Training

During fine-tuning, both clean and perturbed examples are fed to the model. The loss function penalizes wrong predictions on attacked inputs, forcing the model to learn **invariant representations** rather than surface-level cues.

- Smooths the decision boundary near adversarial regions
- Reduces over-reliance on single high-frequency tokens
- Improves generalization to unseen attack variants
- **Trade-off:** Training on a fixed attack set may slightly hurt performance on unseen attack types (observed with Adv V2 on some conditions)

### Input Preprocessing / Normalization

Text is cleaned before inference: spelling correction, Unicode normalization, punctuation standardization, and whitespace cleanup.

- Neutralizes typo-based attacks before tokenization
- Reduces `[UNK]` tokens caused by misspellings
- Zero training cost — pure inference-time defense
- **Limitation:** Has no effect on semantically valid attacks (negation, POS synonym, phrase insertion)

### Baseline vs. Defended (DistilBERT, Accuracy)

| Attack | Baseline | Attacked | Defended (Adv Basic) | Δ |
|---|---|---|---|---|
| Dropout | 0.8500 | 0.7967 | 0.8100 | +1.33% |
| Synonym | 0.8500 | 0.8200 | 0.8367 | +1.67% |
| Typo | 0.8500 | 0.8300 | 0.8467 | +1.67% |
| Negation | 0.8500 | 0.8300 | 0.8433 | +1.33% |
| POS Synonym | 0.8500 | 0.8300 | 0.8433 | +1.33% |
| Phrase | 0.8500 | 0.8433 | 0.8500 | +0.67% |

---

## Final Insights & Conclusion

1. **Adversarial training is the most effective defense** — even a basic variant trained on a subset of attacks outperforms all other configurations on overall robustness score.

2. **Model size is not a proxy for robustness** — RoBERTa's larger pre-training helps clean accuracy but does not translate to proportionally better adversarial robustness.

3. **The accuracy–robustness trade-off is real** — DistilBERT Adv V2 achieves the highest clean accuracy (0.8733) yet scores the lowest on robustness. Engineers must decide which matters more for their deployment context.

4. **Tokenization is a hidden vulnerability** — sub-word tokenizers amplify character-level attacks far beyond what the perturbation itself would suggest. Hardening tokenization (e.g., byte-level BPE) is an underexplored defense direction.

5. **Negation remains an open problem** — no evaluated model reliably handles polarity inversion. This points to a structural limitation in how transformers weight syntactic relationships versus lexical salience.

6. **Preprocessing is worth deploying regardless** — it's free, handles the most common real-world noise (user typos, OCR errors), and stacks with adversarial training.

---

## How to Run

**Requirements:** Google Colab (T4 GPU recommended) or any Python 3.8+ environment with GPU access.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/transformer-robustness-eval.git

# 2. Open the notebook
# Upload to Google Colab or run locally with Jupyter

# 3. Install dependencies (first cell handles this)
pip install transformers datasets torch scikit-learn

# 4. Run all cells sequentially
# Steps 1–6 execute the full pipeline:
#   Step 1: Model setup and dataset loading
#   Step 2: Adversarial attack generation
#   Step 3: Multi-model evaluation
#   Step 4: Defense mechanism training
#   Step 5: Failure analysis and categorization
#   Step 6: Final report and comparison tables
```

**Colab:** Runtime → Run All. Estimated time: ~15–20 minutes on T4 GPU.

---

## Project Structure

```
transformer-robustness-eval/
│
├── Robustness_Analysis_of_Transformer_Models_under_Adversarial_Attacks.ipynb
├── README.md

```

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `Google Colab (T4 GPU)`

---

*Built as a systematic robustness evaluation framework — designed to surface real failure modes, not just report benchmark numbers.*
