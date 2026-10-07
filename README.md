[README_MSCF-DM.md](https://github.com/user-attachments/files/33160134/README_MSCF-DM.md)
# MSCF-DM

Official implementation of **MSCF-DM: Center-Site-Aware Attention and Gated Multi-Scale Fusion Network for Cross-Species DNA Methylation Site Prediction**.

MSCF-DM is a multi-species DNA methylation site prediction framework for three modification types: **4mC, 5hmC, and 6mA**. The model integrates multi-granularity sequence encoding, enhanced nucleotide composition (ENAC), positional information, species information, residual multi-scale dilated depthwise-separable convolution, BiGRU, species-conditioned feature modulation (Species-FiLM), feature-gated fusion, and center-aware attention pooling.

The benchmark used in the study contains **17 modification-type–species datasets**. The three modification types are modeled as three independent prediction tasks, while species belonging to the same modification type are jointly modeled.

---

## Repository structure

The uploaded repository package has the following structure:

```text
MSCF-DM/
├── feature_extraction.py
├── model.py
├── model_train_test.py
└── dataset/
    ├── 4mC-all/
    │   ├── C.equisetifolia-*_neg.txt
    │   ├── C.equisetifolia-*_pos.txt
    │   ├── F.vesca-*_neg.txt
    │   ├── F.vesca-*_pos.txt
    │   ├── S.cerevisiae-*_neg.txt
    │   ├── S.cerevisiae-*_pos.txt
    │   ├── Ts. SUP5-1-*_neg.txt
    │   └── Ts. SUP5-1-*_pos.txt
    ├── 5hmC-all/
    │   ├── H. sapiens-*_neg.fasta
    │   ├── H. sapiens-*_pos.fasta
    │   ├── M. musculus-*_neg.fasta
    │   └── M. musculus-*_pos.fasta
    └── 6mA-all/
        ├── A. thaliana-*_neg.txt / *_pos.txt
        ├── C. elegans-*_neg.txt / *_pos.txt
        ├── C. equisetifolia-*_neg.txt / *_pos.txt
        ├── D. melanogaster-*_neg.txt / *_pos.txt
        ├── F. vesca-*_neg.txt / *_pos.txt
        ├── H. sapiens-*_neg.txt / *_pos.txt
        ├── R. chinensis-*_neg.txt / *_pos.txt
        ├── S. cerevisiae-*_neg.txt / *_pos.txt
        ├── T. thermophila-*_neg.txt / *_pos.txt
        ├── Ts. SUP5-1-*_neg.txt / *_pos.txt
        └── Xoc. BLS256-*_neg.txt / *_pos.txt
```

Here, `*` denotes either `train` or `test`. Positive and negative samples are stored separately.

---

## Main files

### `feature_extraction.py`
Contains the feature-encoding and preprocessing components used to convert DNA sequences into model inputs.

### `model.py`
Contains the implementation of the MSCF-DM network architecture.

### `model_train_test.py`
Main script for model training and evaluation on the benchmark datasets.

---

## Benchmark datasets

The benchmark contains 17 modification-type–species datasets:

- **4mC:** 4 datasets
- **5hmC:** 2 datasets
- **6mA:** 11 datasets

All DNA fragments have a fixed length of **41 bp**, with the candidate site located at the center of the sequence. For 4mC and 5hmC, the central nucleotide is cytosine (C); for 6mA, the central nucleotide is adenine (A).

The supplied files already contain fixed training and independent test partitions. To ensure comparability with the reported experiments, these partitions should be kept unchanged.

The benchmark datasets were originally processed using CD-HIT with an **80% sequence-identity threshold** to reduce sequence redundancy and potential homology bias, following the original benchmark construction protocol.

---

## Environment

MSCF-DM is implemented in **Python** using **PyTorch**.

A typical setup is:

```bash
git clone https://github.com/weixin7112/MSCF-DM.git
cd MSCF-DM
```

Create and activate a Python environment, then install PyTorch and the packages required by the imports in the source files. The uploaded package used to prepare this README does not contain a `requirements.txt` file, so exact dependency versions are not encoded in the archive.

Example:

```bash
conda create -n mscfdm python=3.10
conda activate mscfdm
pip install torch numpy scikit-learn
```

If additional import errors occur, install the corresponding packages required by the scripts.

---

## Training and evaluation

The provided package uses `model_train_test.py` as the main training/evaluation entry script.

From the repository root, run:

```bash
python model_train_test.py
```

Before running, make sure that the dataset path and task/model settings used by `model_train_test.py` point to the corresponding dataset directory:

```text
dataset/4mC-all/
dataset/5hmC-all/
dataset/6mA-all/
```

The three modification types are trained independently.

### Main training settings used in the manuscript

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
| Species-FiLM modulation strength (λ) | 0.2 |
| Gaussian position-feature σ values | 2, 4, 8 |
| Center-prior σ | 4 |

Model selection is performed within the training data, while the independent test set is used for final performance evaluation.

---

## Reproducibility and random seeds

To assess training stability, the experiments were repeated using five random seeds:

```text
2026, 3047, 5179, 6911, 8243
```

For reproducibility, keep the train/test partitions fixed and change only the random seed between repeated runs.

The evaluation metrics used in the study include:

- Accuracy (ACC)
- Sensitivity / Recall (SN)
- Specificity (SP)
- Matthews correlation coefficient (MCC)
- Area under the ROC curve (AUC)
- F1 score

---

## Model overview

MSCF-DM contains the following main components:

1. **Multi-granularity sequence encoding** using nucleotide/k-mer representations.
2. **ENAC and positional features** for local composition and candidate-site position information.
3. **Species information encoding** for species-aware prediction within each modification type.
4. **MS-D-DSConv** for multi-scale local sequence-pattern extraction.
5. **BiGRU** for bidirectional contextual dependency modeling.
6. **Species-FiLM** for species-conditioned modulation of convolutional and recurrent features.
7. **Feature-gated fusion** for adaptive integration of the two feature branches.
8. **Center-aware attention pooling** for emphasizing candidate-site-related information.

The shared feature dimension is 128, the BiGRU hidden size in each direction is 64, and the species embedding dimension is 32.

---

## Computational profile

MSCF-DM is designed as a lightweight model and provides a favorable balance between predictive performance and computational efficiency. The revised study reports approximately **0.440 M trainable parameters** and **0.0213 G FLOPs** under the adopted complexity-counting procedure.

---

## Pretrained weights and end-to-end prediction

For direct reproduction and independent verification, it is recommended that the public repository provide:

```text
checkpoints/
├── 4mC/
├── 5hmC/
└── 6mA/
```

with trained checkpoints for the three prediction tasks, together with an end-to-end prediction script that loads the corresponding checkpoint and accepts user-provided 41-bp sequences.

**Important:** the archive supplied for preparation of this README contains the three source-code files and benchmark datasets listed above, but it does **not** contain pretrained checkpoint files or a separate end-to-end prediction script. If these files are already present in the online GitHub repository, add their exact filenames and usage commands to this section before publication.

---

## Citation

If you use MSCF-DM in your research, please cite the corresponding paper:

```bibtex
@article{MSCFDM,
  title   = {MSCF-DM: Center-Site-Aware Attention and Gated Multi-Scale Fusion Network for Cross-Species DNA Methylation Site Prediction},
  author  = {Wei, Xin and Hu, Siqin},
  journal = {To be updated},
  year    = {2026}
}
```

Please update the journal information, volume, pages, DOI, and publication year after publication.

---

## Contact

For questions regarding the code or experiments, please contact the authors through the repository or the contact information provided in the manuscript.

Repository: https://github.com/weixin7112/MSCF-DM
