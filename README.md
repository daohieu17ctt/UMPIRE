# UMPIRE: Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

**Official repository for the paper: _Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume_** [[Paper]](https://arxiv.org/abs/2602.24195)  

**Abstract:** Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. **We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features.** UMPIRE computes the incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, effectively capturing both the global semantic diversity of samples and the local incoherence of responses based on internal model confidence. We propose uncertainty desiderata for MLLMs and provide theoretical analysis motivating UMPIRE's design. Extensive experiments show that UMPIRE consistently outperforms baseline metrics in error detection and uncertainty calibration across image, audio, and video-text benchmarks, including adversarial and out-of-distribution settings. We also demonstrate UMPIRE's generalization to non-text output tasks, including image and audio generation.

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

Earlier versions of our work were presented at the [ICLR 2025 Quantify Uncertainty and Hallucination in Foundation Models (QUESTION) Workshop](https://datafm.github.io/) in Mar 2025 and the [ICML 2025 Workshop on Reliable and Responsible Foundation Models (R2-FM'25)](https://r2-fm.github.io/).

Please cite our paper:

```bibtex
@article{lau2026uncertainty,
  title={Uncertainty quantification for multimodal large language models with incoherence-adjusted semantic volume},
  author={Lau, Gregory Kang Ruey and Dao, Hieu and Lin, Nicole Kan Hui and Low, Bryan Kian Hsiang},
  journal={arXiv preprint arXiv:2602.24195},
  year={2026}
}
```

---

## 📬 Contact

For questions or feedback, please open an issue or contact:
[daohieu@comp.nus.edu.sg](mailto:daohieu@comp.nus.edu.sg) or [greglau@comp.nus.edu.sg](mailto:greglau@comp.nus.edu.sg)
