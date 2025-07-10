# HiFuseNet

---

## 📊 Data Availability

The data used in this manuscript are publicly and freely available from the following sources:

- **MoleculeNet benchmark datasets**:  
  [https://moleculenet.org/datasets-1](https://moleculenet.org/datasets-1)

- **Corrosion inhibitor dataset**:  
  [https://www.corrdata.org.cn/inhibitor/](https://www.corrdata.org.cn/inhibitor/)

---

## 📁 Code Availability

More detailed instructions and supplementary details will be available upon formal acceptance of the manuscript.

---
To run the code, We advice the following environment:

| Component       | Version         |
|-----------------|------------------|
| `Python`        | 3.9              |
| `rdkit`         | 2024.3.6         |
| `torch`         | 1.13.1+cu117     |
| `scikit-learn`  | 1.5.2            |
| `numpy`         | 1.23.0           |
| `pandas`        | 1.2.0            |


## 🔍 Prediction Usage

We provide an inference script `predict.py` that can perform property prediction on new molecules using pre-trained models (`model.pth`) saved from different cross-validation folds.

### ✅ Script: `predict.py`

This script supports prediction using 10 saved models (one per fold), batch inference, and automatic evaluation if ground truth is provided.



### 🧩 Inputs

- `--property`: Name of the dataset (e.g., `corrosion`, `HIV`, etc.). This is used to locate the saved models under `./data/{property}/fold_*/model.pth`.
- `--smiles`: Path to a CSV file containing the SMILES strings to predict.  
  - **First column**: SMILES string  
  - **Second column (optional)**: Ground truth label(s), used to compute RMSE
- `--transformer`: Path to a `.npy` file containing 512-dimensional SMILES-based transformer features (same order as SMILES in CSV).
- `--output`: Path to save the prediction results in CSV format.
- `--batch_size` (optional): Batch size for inference. Default: `32`.



### 📤 Output

The output CSV file will contain:

- Original SMILES and (if provided) ground truth column(s)
- 10 new columns: `fold_0`, `fold_1`, ..., `fold_9` — predictions from each fold model
- (Optional) RMSE printed to console if ground truth exists



### 🧪 Example

```bash
python predict.py \
  --property corrosion \
  --smiles ./example.csv \
  --transformer ./example.npy \
  --output ./predicted_example.csv \
  --batch_size 32
