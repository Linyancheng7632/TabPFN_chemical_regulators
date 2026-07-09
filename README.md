# Project Hormones — Machine Learning-Based Prediction of Fatty Acid Yields in *Schizochytrium* Fermentation

This project uses multiple ML regression models (XGBoost, RandomForest, SVM, LightGBM, CatBoost, TabPFN) to predict EPA titer, EPA percentage, and lipid titer from chemical elicitor concentrations.

---

## Directory Structure

```
project_hormones/
├── data/
│   ├── train.csv                     # Training data
│   └── predict.csv                   # New data for prediction
├── model/
│   └── tabpfn-v2-regressor.ckpt      # TabPFN v2 regressor weights (~42.4 MB)
├── main.py                           # Main training / testing script
├── model_selection.py                # Optuna-based hyperparameter search
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n tabpfn python=3.12
conda activate tabpfn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. TabPFN Model Weights

The TabPFN v2 regressor checkpoint is included at `model/tabpfn-v2-regressor.ckpt`. If the file is missing, download it from HuggingFace:

```bash
python -c "from tabpfn import TabPFNRegressor; TabPFNRegressor()"
```

---

## How to Run

### Before Running — Edit These Configurations

The code contains hardcoded paths from the original Linux server. Change the following in both `main.py` and `model_selection.py` before running.

**Project root path** — replace `/home/m24linyc/DL/project_hormones/` with your local directory:

- `main.py` line 27: `config['root_project'] = 'YOUR_PROJECT_PATH/'`
- `model_selection.py` line 31: `config['root'] = 'YOUR_PROJECT_PATH/'`

**TabPFN model path** — point to the local checkpoint:

- `main.py` line 316: `model_path="YOUR_PROJECT_PATH/model/tabpfn-v2-regressor.ckpt"`
- `model_selection.py` line 47: `config['tabpfn_model_path'] = 'YOUR_PROJECT_PATH/model/tabpfn-v2-regressor.ckpt'`

**Compute device** — `main.py` line 317:

- With NVIDIA GPU: keep `device="cuda:0"` (or your GPU index)
- Without GPU: change to `device="cpu"`

> `model_selection.py` reads the device automatically from `--gpu_ids`. The trailing `/` on project paths is required.

### Run

```bash
python main.py
```

Mode, target label, and model are set via the `config` dictionary in `main.py` (around line 40).

### model_selection.py — Hyperparameter Search

```bash
python model_selection.py
```

Target label, model, K-folds, and number of Optuna trials are set via the `config` dictionary in `model_selection.py` (around line 30). Optimization objective: maximize Spearman correlation. Best parameters are logged to `results/logs_ml/{timestamp}/out.log`.
