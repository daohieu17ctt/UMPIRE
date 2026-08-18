# UMPIRE: Uncertainty Quantification for Multimodal Large Language Models

**Official repository for the paper: _Uncertainty Quantification for Multimodal Large Language Models_** [[Paper]](https://openreview.net/pdf?id=2UYZHvXUAH)  

**Abstract:** Multimodal Large Language Models (MLLMs) hold promise in tackling challenging multimodal tasks, but may generate seemingly plausible but erroneous output, making them hard to trust and deploy in real-life settings. Generating accurate uncertainty metrics quickly for each MLLM response during inference could enable interventions such as escalating queries with uncertain responses to human experts or larger models for improved performance. However, existing uncertainty quantification methods require external verifiers, additional training, or high computational resources, and struggle to handle scenarios such as out-of-distribution (OOD) or adversarial settings. To overcome these limitations, **we present UMPIRE, an efficient and effective training-free framework to estimate MLLM output uncertainty at inference time without external tools**, by computing metrics based on the diversity of the MLLM’s responses that is augmented with internal indicators of each output’s coherence. We empirically show that our method significantly outperforms benchmarks in predicting incorrect responses and providing calibrated uncertainty estimates, including for OOD, adversarial and domain-specific (e.g., medical radiology) data settings.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/daohieu17ctt/UMPIRE.git
cd UMPIRE
```

### 2. Environment Setup

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Or with `conda`:

```bash
conda env create -f environment.yml
conda activate umpire
```

### 3. Data Preparation

Ensure your datasets (OKVQA, VQAv2, AdVQA) are placed under the `data/` directory in their respective subfolders. If preprocessing is needed, change the question-answer json file path in this script and run it:

```bash
bash scripts/preprocess_data.sh
```

Please note that this script is only used for the VQAv2-format datasets (such as OKVQA, VQAv2, AdVQA), you need to preprocess your own dataset following the format in ```pipeline/preprocess_data.py```

Please download the image for each dataset and prepare the image directory path for the next step. Note that VQAv2 and OKVQA use [COCO-val2014 split](http://images.cocodataset.org/zips/val2014.zip) while AdVQA uses [COCO-val2017 split](http://images.cocodataset.org/zips/val2017.zip).

### 4. Generate Embeddings & Evaluate

```bash
# Step 1: Generate responses, embeddings and log-likelihoods
bash scripts/generate_and_compute_embedding.sh

# Step 2: Compute UMPIRE (and the baselines) and evaluate
bash scripts/compute_umpire_and_evaluate.sh
```

Step 2 reports AUROC, calibrated ECE, Pearson correlation, TPR at fixed FPR, and AURAC for
UMPIRE alongside three baselines (length-normalized entropy, semantic entropy, eigenscore),
and writes them to `<output_dir>/umpire_results.json`. Semantic entropy needs a GPU for its
DeBERTa entailment model; pass `--re_cluster_semantic_entropy` to re-cluster the responses
from scratch rather than reusing pre-computed `cluster_ids` (this takes a few hours).

**Results for the OKVQA dataset** (`llava-v1.5-13b`, 50 generations per prompt):

|                  |   auc |   cece |   pearsonr |   tpr_at_0.1_fpr |   tpr_at_0.01_fpr |   aurac |
|:-----------------|------:|-------:|-----------:|-----------------:|------------------:|--------:|
| ln_entropy       | 0.704 |  0.044 |      0.851 |            0.242 |             0.03  |   0.789 |
| semantic_entropy | 0.714 |  0.143 |      0.251 |            0.322 |             0.053 |   0.772 |
| eigen_score      | 0.737 |  0.161 |      0.894 |            0.332 |             0.075 |   0.802 |
| umpire           | **0.754** |  **0.042** |      **0.964** |            **0.365** |             **0.090** |   **0.808** |

### 5. Single-example demo

To score one image-question pair directly:

```bash
bash demo/demo.sh
```

---

## 📚 Citation

Our work was first accepted at the [ICLR 2025 Quantify Uncertainty and Hallucination in Foundation Models (QUESTION) Workshop](https://datafm.github.io/) in Mar 2025, and an expended version was accepted at the [ICML 2025 Workshop on Reliable and Responsible Foundaation Models (R2-FM’25)](https://r2-fm.github.io/).

Please cite our paper:

```bibtex
@inproceedings{
lau2025uncertainty,
title={Uncertainty Quantification for {MLLM}s},
author={Gregory Kang Ruey Lau and Hieu Dao and Nicole Kan Hui Lin and Bryan Kian Hsiang Low},
booktitle={ICML 2025 Workshop on Reliable and Responsible Foundation Models},
year={2025},
url={https://openreview.net/forum?id=2UYZHvXUAH}
}
```

---

## 📬 Contact

For questions or feedback, please open an issue or contact:
[daohieu@comp.nus.edu.sg](mailto:daohieu@comp.nus.edu.sg) or [greglau@comp.nus.edu.sg](mailto:greglau@comp.nus.edu.sg)
