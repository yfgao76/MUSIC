# MUSIC
Code for our ICWSM2025 smm4h workshop paper.


## Abstract
SMM4H 2025 Task 1 requires detecting adverse drug‑event (ADE) mentions in social‑media posts of four languages. We introduce MUSIC, a compact two‑stage decoder-only language model to tackle the task. MUSIC combines a Classifier that assigns an initial label with a Judge that confirms or overturns that decision. An ensemble of Classifier and Judge checkpoints achieves a weighted-F1 of 0.7079 on the blind test set, outperforming the task median by 8 percentage points and the mean by 17.

## Get Started

**Installation**

```bash
conda create -n music python=3.12
conda activate music
pip install -r requirements.txt
```

**Inference**

You need to modify model path and dataset path to make inference. Sample datasets are provided in the repository. You can download our sample pretraining weight on huggingface: [MUSIC_pretrained](https://huggingface.co/infmourir/MUSIC_pretrained)

```bash 
# inference on classifier model
python classifier_inference.py
# inference on judge model
python judge_inference.py
```

**Fine-tune your own model**

You can use trainer scripts to fine-tune your own model.

