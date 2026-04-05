# Animal image classification: CNN vs. transfer learning

PyTorch notebook for **10-class animal image classification**. It compares:

1. A **CNN trained from scratch**
2. **Frozen transfer learning** — ResNet-50 as a fixed feature extractor with a trainable classifier
3. **Fine-tuned transfer learning** — ResNet-50 with the last block and classifier updated for the target dataset

The workflow also includes **exploratory analysis** (e.g. PCA, UMAP, clustering) alongside training and evaluation.

## Contents

| File | Description |
|------|-------------|
| `CNN.ipynb` | End-to-end pipeline: dataset exploration, training, validation, and comparison of the three modeling strategies |

## Requirements

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) and `torchvision`  
- NumPy, pandas, matplotlib, Pillow  
- scikit-learn  
- [umap-learn](https://pypi.org/project/umap-learn/)  
- tqdm  

Example (CPU-only PyTorch; adjust install from [pytorch.org](https://pytorch.org/) if you use CUDA):

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn umap-learn tqdm
```

## Where to get the data

The splits in this project match the **Animals-10** benchmark: ten classes (`butterfly`, `cat`, `chicken`, `cow`, `dog`, `elephant`, `horse`, `sheep`, `spider`, `squirrel`).

1. **Public download (recommended)**  
   Use **[Animals-10 on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)** (free account required). The archive is usually organized as **one folder per class**. You must **build your own `train` / `val` / `test` split** (for example ~80% / 10% / 10% per class, stratified) and place them under `animal_groups/` as below. Image counts do not need to match the notebook exactly; the code only expects **10 classes** and the three splits.

2. **Course or shared bundle**  
   If you received this notebook for a class (paths like `163602/datasets` suggest a course layout), use the **dataset files supplied by the instructor** or LMS — that is the intended copy for assignments.

3. **Other sources**  
   Any image collection with the **same 10 class folders** will work if you split it the same way. Always check the **license and terms** of the dataset you use.

## Data layout

The notebook expects a root such as:

`163602/datasets/animal_groups/`

with this structure:

```text
animal_groups/
├── train/
│   ├── butterfly/
│   ├── cat/
│   └── ... (one folder per class)
├── val/
│   └── (same class folder names)
└── test/
    └── (same class folder names)
```

Each split contains one subdirectory per class. If your data lives elsewhere, update `DATA_ROOT` (and related paths) in the first configuration cells of `CNN.ipynb`.

## Running

1. Clone or download this repository.  
2. Place the dataset as above (or change paths in the notebook).  
3. Open `CNN.ipynb` in **Jupyter Lab**, **Jupyter Notebook**, or **VS Code** and run cells **top to bottom**.

```bash
jupyter lab CNN.ipynb
```

## Suggested repository names

| Use case | Examples |
|----------|----------|
| Clear & generic | `pytorch-animal-classification`, `animal-cnn-transfer-learning` |
| Course / assignment | `hw3-animal-classification`, `cs163602-animal-cnn` |

## Acknowledgments

The material covering **frozen ResNet-50 vs. fine-tuned ResNet-50**, with **PCA/UMAP-style exploration**, comes from [unsupervised-clustering-on-wildlife-imaging](https://github.com/alanberdikulov/unsupervised-clustering-on-wildlife-imaging) — wildlife image clustering with ResNet-50 embeddings, PCA, UMAP, and K-Means.