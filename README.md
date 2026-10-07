[README_MSCF-DM.md](https://github.com/user-attachments/files/33148043/README_MSCF-DM.md)
# MSCF-DM

Official implementation of **MSCF-DM: Center-Site-Aware Attention and Gated Multi-Scale Fusion Network for Cross-Species DNA Methylation Site Prediction**.

MSCF-DM is a unified multi-species prediction framework for three DNA methylation types: **4mC, 5hmC, and 6mA**. The model combines multi-granularity sequence encoding, ENAC, positional encoding, species information, multi-scale dilated depthwise-separable convolution, BiGRU, species-conditioned feature modulation (Species-FiLM), feature-gated fusion, and center-aware attention pooling.

The experiments reported in the manuscript were conducted on **17 species-specific benchmark datasets** covering 4mC, 5hmC, and 6mA.

> **Important:** Before publishing this README, make sure that the checkpoint filenames and script paths below match the actual files in this repository. If your current filenames differ, replace them accordingly.

---

## 1. Repository structure

A clear repository structure is recommended as follows:

```text
MSCF-DM/
├── data/                       # Benchmark datasets
├── checkpoints/                # Trained model weights
│   ├── 4mC/
│   │   └── best_model.pt
│   ├── 5hmC/
│   │   └── best_model.pt
│   └── 6mA/
│       └── best_model.pt
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── predict.py                  # End-to-end prediction script
├── requirements.txt            # Python dependencies
└── README.md
```

The three trained checkpoints correspond to the three independently trained prediction tasks:

- `checkpoints/4mC/best_model.pt`
- `checkpoints/5hmC/best_model.pt`
- `checkpoints/6mA/best_model.pt`

---

## 2. Environment

MSCF-DM was implemented in **PyTorch**.

Install the required dependencies with:

```bash
git clone https://github.com/weixin7112/MSCF-DM.git
cd MSCF-DM
pip install -r requirements.txt
```

If CUDA is available, please install a PyTorch build compatible with your CUDA version.

---

## 3. Benchmark datasets

The benchmark contains **17 modification-type/species datasets**:

- **4mC:** 4 datasets
- **5hmC:** 2 datasets
- **6mA:** 11 datasets

All DNA fragments have a fixed length of **41 bp**, with the candidate site located at the center of the sequence.

The three methylation types are trained as **three independent tasks**. Within each task, samples from all species belonging to that modification type are jointly used for training.

The fixed training and independent test partitions used in the manuscript should be placed under `data/`.

Example:

```text
data/
├── 4mC/
├── 5hmC/
└── 6mA/
```

---

## 4. Trained model weights

To facilitate reproducibility, trained model weights for all three DNA modification tasks are provided:

| Task | Checkpoint |
|---|---|
| 4mC | `checkpoints/4mC/best_model.pt` |
| 5hmC | `checkpoints/5hmC/best_model.pt` |
| 6mA | `checkpoints/6mA/best_model.pt` |

These checkpoints can be used directly for prediction or independent-test evaluation without retraining.

---

## 5. End-to-end prediction

The `predict.py` script provides an end-to-end interface for loading a trained checkpoint and predicting DNA methylation sites.

### 5.1 Input

Each input sequence should:

1. have a length of **41 bp**;
2. contain only valid DNA bases (`A`, `C`, `G`, and `T`);
3. be associated with the corresponding species identifier used by the model.

A recommended CSV input format is:

```text
sequence,species
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA,H.sapiens
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT,H.sapiens
```

If the prediction script in this repository uses a different delimiter or column order, follow the example input file provided with the code.

### 5.2 Prediction commands

#### 4mC

```bash
python predict.py \
    --task 4mC \
    --input path/to/input.csv \
    --checkpoint checkpoints/4mC/best_model.pt \
    --output predictions_4mC.csv
```

#### 5hmC

```bash
python predict.py \
    --task 5hmC \
    --input path/to/input.csv \
    --checkpoint checkpoints/5hmC/best_model.pt \
    --output predictions_5hmC.csv
```

#### 6mA

```bash
python predict.py \
    --task 6mA \
    --input path/to/input.csv \
    --checkpoint checkpoints/6mA/best_model.pt \
    --output predictions_6mA.csv
```

The output file should contain at least:

```text
sequence,species,probability,predicted_label
```

where `probability` is the predicted probability of the positive methylation class and `predicted_label` is the binary prediction obtained using the default decision threshold.

---

## 6. Evaluation on the benchmark test sets

To evaluate a trained checkpoint on the corresponding independent test set:

```bash
python evaluate.py \
    --task 4mC \
    --checkpoint checkpoints/4mC/best_model.pt \
    --data data/4mC
```

Replace `4mC` with `5hmC` or `6mA` for the other two tasks.

The evaluation procedure reports the metrics used in the manuscript:

- Accuracy (ACC)
- Sensitivity / Recall (SN)
- Specificity (SP)
- Matthews correlation coefficient (MCC)
- Area under the ROC curve (AUC)
- F1 score

---

## 7. Training from scratch

Example:

```bash
python train.py \
    --task 4mC \
    --data data/4mC \
    --seed 2026
```

The manuscript reports repeated experiments using the following random seeds:

```text
2026, 3047, 5179, 6911, 8243
```

Use the same fixed training/test partition for all seeds.

---

## 8. Main training settings

The main hyperparameters used in the manuscript are:

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Maximum epochs | 50 |
| Early-stopping patience | 10 |
| Dropout | 0.18 |
| Label smoothing | 0.02 |
| Learning-rate schedule | 3 warm-up epochs + cosine annealing |
| Species-FiLM modulation strength (`lambda`) | 0.2 |
| Position-feature sigmas | 2, 4, 8 |
| Center-prior sigma | 4 |

Model selection is performed using five-fold cross-validation within the training data, whereas the independent test set is used only for final evaluation.

---

## 9. Computational complexity

The MSCF-DM model contains approximately:

- **0.440 M trainable parameters**
- **0.0213 G FLOPs** per prediction for a 41-bp input sequence

These values correspond to the computational-complexity analysis reported in the revised supplementary material.

---

## 10. Reproducibility

To reproduce the reported experiments:

1. Use the fixed benchmark train/test partitions provided in `data/`.
2. Select one of the three tasks: `4mC`, `5hmC`, or `6mA`.
3. Train the model using the hyperparameters above.
4. Repeat the experiment with seeds `2026`, `3047`, `5179`, `6911`, and `8243`.
5. Evaluate each trained model on the same independent test set.
6. Report the mean and standard deviation across the five runs.

For direct verification without retraining, use the trained checkpoints provided in `checkpoints/`.

---

## 11. Citation

If you use MSCF-DM in your research, please cite the corresponding paper:

```bibtex
@article{MSCFDM,
  title   = {MSCF-DM: Center-Site-Aware Attention and Gated Multi-Scale Fusion Network for Cross-Species DNA Methylation Site Prediction},
  author  = {Wei, Xin and Hu, Siqin},
  journal = {To be updated},
  year    = {2026}
}
```

Please update the journal, volume, pages, DOI, and publication year after publication.

---

## 12. Contact

For questions regarding the code or experiments, please contact:

- **Xin Wei**: jci_wx@163.com
- **Siqin Hu**: husiqin1004@163.com

Repository: https://github.com/weixin7112/MSCF-DM
