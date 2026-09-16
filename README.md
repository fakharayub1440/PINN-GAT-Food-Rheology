# Physics-Informed Graph Attention Networks for Food Hydrocolloid Rheology

Official codebase and benchmark reproduction suite for the manuscript:  
**"Predicting Non-Linear Rheology of Food Hydrocolloids for Process Engineering via Physics-Informed Graph Attention Networks: A Molecular Topology Benchmark Study"**  
Submitted to the *Journal of Food Engineering* (Manuscript No.: `JFOODENG-D-26-02541`).

---

## 1. Overview

Predicting the non-linear rheology of food hydrocolloids directly from chemical structure addresses key formulation challenges in food process engineering. This framework establishes an algorithmic benchmark that maps 2D molecular graphs of biopolymers directly to fluid viscosity across operational coordinates (concentration $C$, temperature $T$, and shear rate $\gamma$) under thermodynamic regularizations.

### Core Architecture
* **Native Graph Formulation:** Extracts 6-dimensional node feature vectors ($Z$, degree, hybridization, aromaticity, formal charge, total valence) directly from canonical SMILES strings using RDKit.
* **Identity-Isolated Partitioning:** Implements strict 5-Fold GroupKFold cross-validation where an entire biopolymer family is completely withheld during testing, eliminating data leakage.
* **Multi-Physics Regularization:** Uses PyTorch reverse-mode automatic differentiation (`torch.autograd.grad`) to enforce thermodynamic monotonic thermal thinning ($\partial\hat{\eta}/\partial T \le 0$).

---

## 2. Experimental Benchmark Results

### A. GNN Architecture Comparison (5-Fold GroupKFold)
Evaluated across five independent held-out hydrocolloid families (Xanthan Gum, Sodium Alginate, Pectin, Kappa-Carrageenan, Guar Gum):

| Model Architecture | Input Topology | 5-Fold Mean $R^2$ | 5-Fold Mean MAE (cp) | Key Architectural Properties |
| :--- | :--- | :--- | :--- | :--- |
| **PINN-GIN** | Molecular Graph | **0.8969** | **120.09** | Weisfeiler-Lehman isomorphism; isotropic aggregation |
| **PINN-Transformer** | Molecular Graph | 0.8716 | 168.21 | Self-attention over local neighborhoods |
| **PINN-GAT (Selected)** | Molecular Graph | **0.8591** | **144.38** | Anisotropic attention ($\alpha_{ij}$); direct XAI explainability |
| **PINN-MPNN** | Molecular Graph | 0.8582 | 181.60 | Message passing conditioned on bond attributes |

PINN-GIN yields slightly higher statistical scores, but PINN-GAT computes directional edge attention weights ($\alpha_{ij}$) across molecular bonds. These coefficients are required for Explainable AI (XAI) feature attribution and chemical group mapping.

### B. Thermodynamic Regularization ($\lambda$) Sensitivity Sweep
Evaluated on the held-out validation fold (Fold 1: Kappa-Carrageenan):

| Physics Weight ($\lambda$) | Validation $R^2$ | MAE (cp) | Thermal Violation Rate (%) | Optimization Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **0.00 (Unconstrained)** | 0.9531 | 102.58 | 0.00% | Fast convergence; unregularized gradient path |
| **0.01** | 0.9522 | 76.61 | 0.00% | Regularized gradient path; 25.3% error reduction |
| **0.05** | 0.9372 | 84.58 | 0.00% | Stable parameter trajectories |
| **0.10 (Baseline)** | 0.9355 | 85.02 | 0.00% | Balanced objective trade-off |
| **0.50** | 0.9544 | 79.44 | 0.00% | Strict derivative bounding |
| **1.00** | **0.9767** | **56.75** | 0.00% | Lowest MAE; 44.7% total error suppression |

---

## 3. Repository Structure

```text
PINN-GAT-Food-Rheology/
├── 01_Data/
│   └── hydrocolloid_smiles.csv       # Curated biopolymer SMILES strings
├── figures/                          # High-resolution architectural schematics
│   ├── figure1_architecture.png
│   └── graphical_abstract.png
├── results/                          # Benchmark outputs & validation records
│   ├── benchmark_gnn_models.csv
│   └── benchmark_lambda_sweep.csv
├── train_benchmark.py                # End-to-end training & evaluation pipeline
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # Documentation
# 1. Clone repository
git clone https://github.com/fakharayub1440/PINN-GAT-Food-Rheology.git
cd PINN-GAT-Food-Rheology

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
python train_benchmark.py
5. Dataset DetailsThe dataset contains canonical SMILES strings for 5 standard food-grade hydrocolloids in 01_Data/hydrocolloid_smiles.csv:Sodium Alginate: O=C(O)C1OC(O)C(O)C(O)C1OPectin: CC(=O)OC1C(O)C(OC)OC(C(=O)O)C1OKappa-Carrageenan: OS(=O)(=O)OC1C(O)C(OC2OC(CO)C(O)C(O)C2O)OC(CO)C1OGuar Gum: OCC1OC(O)C(O)C(O)C1OC2OC(CO)C(O)C(O)C2OXanthan Gum: CC1OC(OC2C(CO)OC(O)C(O)C2O)C(O)C(O)C1OOperational parameters span:Concentration ($C$): $0.5\text{ wt\%} \le C \le 5.0\text{ wt\%}$Temperature ($T$): $20.0^\circ\text{C} \le T \le 80.0^\circ\text{C}$Shear Rate ($\gamma$): $1.0\text{ s}^{-1} \le \gamma \le 100.0\text{ s}^{-1}$
@article{ayub2026predicting,
  title={Predicting Non-Linear Rheology of Food Hydrocolloids for Process Engineering via Physics-Informed Graph Attention Networks: A Molecular Topology Approach},
  author={Ayub, Fakhar and collaborators},
  journal={Journal of Food Engineering},
  volume={Under Review},
  year={2026},
  publisher={Elsevier}
}
