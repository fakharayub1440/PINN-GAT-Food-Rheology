import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GINConv,
    GraphNorm,
    NNConv,
    TransformerConv,
    global_add_pool,
)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)
os.makedirs("01_Data", exist_ok=True)

# Curated SMILES Representation of Hydrocolloids
polymers = {
    "Kappa-Carrageenan": "OS(=O)(=O)OC1C(O)C(OC2OC(CO)C(O)C(O)C2O)OC(CO)C1O",
    "Sodium Alginate": "O=C(O)C1OC(O)C(O)C(O)C1O",
    "Guar Gum": "OCC1OC(O)C(O)C(O)C1OC2OC(CO)C(O)C(O)C2O",
    "Xanthan Gum": "CC1OC(OC2C(CO)OC(O)C(O)C2O)C(O)C(O)C1O",
    "Pectin": "CC(=O)OC1C(O)C(OC)OC(C(=O)O)C1O",
}


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features = [
        [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetHybridization()),
            1.0 if atom.GetIsAromatic() else 0.0,
            float(atom.GetFormalCharge()),
            float(atom.GetTotalValence()),
        ]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        b_type = bond.GetBondTypeAsDouble()
        edge_attrs.extend([[b_type], [b_type]])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return x, edge_index, edge_attr, mol.GetNumAtoms()


rows, graphs_cache = [], {}
for name, smi in polymers.items():
    x, e_idx, e_att, n_atoms = smiles_to_graph(smi)
    graphs_cache[name] = (x, e_idx, e_att, n_atoms)
    c = np.random.uniform(0.5, 5.0, 500)
    t = np.random.uniform(20.0, 80.0, 500)
    gamma = np.random.uniform(1.0, 100.0, 500)
    viscosity = (n_atoms * 15.0 * (c**1.8)) / (
        np.log10(t) * np.log10(gamma + 1.0)
    )
    for i in range(500):
        rows.append(
            {
                "polymer": name,
                "C": c[i],
                "T": t[i],
                "gamma": gamma[i],
                "viscosity": viscosity[i],
            }
        )

df = pd.DataFrame(rows)


class ProcessMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
        )

    def forward(self, p):
        return self.net(p)


class Regressor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, f):
        return self.net(f)


class GenericPINN_GNN(nn.Module):

    def __init__(self, gnn_type="GAT"):
        super().__init__()
        self.gnn_type = gnn_type
        self.proj = nn.Linear(6, 32)
        self.norm = GraphNorm(32)
        if gnn_type == "GAT":
            self.gnn1 = GATConv(32, 32, heads=2, concat=False)
            self.gnn2 = GATConv(32, 64, heads=1, concat=False)
        elif gnn_type == "GIN":
            self.gnn1 = GINConv(
                nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
            )
            self.gnn2 = GINConv(
                nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
            )
        elif gnn_type == "Transformer":
            self.gnn1 = TransformerConv(32, 32, heads=2, concat=False)
            self.gnn2 = TransformerConv(32, 64, heads=1, concat=False)
        elif gnn_type == "MPNN":
            nn1 = nn.Sequential(
                nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 32 * 32)
            )
            nn2 = nn.Sequential(
                nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 32 * 64)
            )
            self.gnn1 = NNConv(32, 32, nn1, aggr="mean")
            self.gnn2 = NNConv(32, 64, nn2, aggr="mean")
        self.process_mlp = ProcessMLP()
        self.regressor = Regressor()

    def forward(self, x, edge_index, edge_attr, batch, proc_vars):
        h = self.norm(self.proj(x))
        if self.gnn_type == "MPNN":
            h = torch.relu(self.gnn1(h, edge_index, edge_attr))
            h = torch.relu(self.gnn2(h, edge_index, edge_attr))
        else:
            h = torch.relu(self.gnn1(h, edge_index))
            h = torch.relu(self.gnn2(h, edge_index))
        x_g = global_add_pool(h, batch)
        p_p = self.process_mlp(proc_vars)
        f = torch.cat([x_g, p_p, x_g * p_p], dim=-1)
        return self.regressor(f)


def train_and_eval(
    model,
    train_loader,
    test_loader,
    target_scaler,
    proc_scaler,
    lambda_phys=0.1,
    epochs=60,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.002, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    mse_criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for data in train_loader:
            data = data.to(device)
            proc_vars = data.proc.clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            pred = model(
                data.x, data.edge_index, data.edge_attr, data.batch, proc_vars
            )
            l_mse = mse_criterion(pred, data.y)
            grads = torch.autograd.grad(
                outputs=pred,
                inputs=proc_vars,
                grad_outputs=torch.ones_like(pred),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            l_phys = torch.mean(torch.relu(grads[:, 1]) ** 2)
            loss = l_mse + lambda_phys * l_phys
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    y_true_list, y_pred_list = [], []
    violations, total_points = 0, 0
    with torch.set_grad_enabled(True):
        for data in test_loader:
            data = data.to(device)
            proc_vars = data.proc.clone().detach().requires_grad_(True)
            pred = model(
                data.x, data.edge_index, data.edge_attr, data.batch, proc_vars
            )
            grads = torch.autograd.grad(
                outputs=pred,
                inputs=proc_vars,
                grad_outputs=torch.ones_like(pred),
                create_graph=False,
                retain_graph=False,
            )[0]
            violations += torch.sum(grads[:, 1] > 1e-4).item()
            total_points += len(grads[:, 1])
            y_pred_orig = target_scaler.inverse_transform(
                pred.detach().cpu().numpy()
            )
            y_true_orig = target_scaler.inverse_transform(data.y.cpu().numpy())
            y_true_list.extend(y_true_orig.flatten())
            y_pred_list.extend(y_pred_orig.flatten())
    return (
        r2_score(y_true_list, y_pred_list),
        mean_absolute_error(y_true_list, y_pred_list),
        (violations / total_points) * 100.0,
    )


def build_pyg_data(sub_df, proc_scaler=None, target_scaler=None, fit=False):
    proc = sub_df[["C", "T", "gamma"]].values
    y = sub_df[["viscosity"]].values
    if fit:
        proc_scaler = MinMaxScaler().fit(proc)
        target_scaler = StandardScaler().fit(y)
    proc_scaled = proc_scaler.transform(proc)
    y_scaled = target_scaler.transform(y)
    data_list = []
    for idx, row in enumerate(sub_df.itertuples()):
        x, e_idx, e_att, _ = graphs_cache[row.polymer]
        d = Data(
            x=x,
            edge_index=e_idx,
            edge_attr=e_att,
            proc=torch.tensor(proc_scaled[idx], dtype=torch.float).unsqueeze(0),
            y=torch.tensor(y_scaled[idx], dtype=torch.float).unsqueeze(0),
        )
        data_list.append(d)
    return data_list, proc_scaler, target_scaler


gkf = GroupKFold(n_splits=5)
folds_data = []
for train_idx, test_idx in gkf.split(df, groups=df["polymer"]):
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    train_data, p_sc, t_sc = build_pyg_data(train_df, fit=True)
    test_data, _, _ = build_pyg_data(
        test_df, proc_scaler=p_sc, target_scaler=t_sc, fit=False
    )
    folds_data.append(
        (test_df["polymer"].iloc[0], train_data, test_data, p_sc, t_sc)
    )

print("=" * 80)
print("EXPERIMENT 1: GNN Architectural Comparison (5-Fold GroupKFold)")
print("=" * 80)

gnn_results = []
for g_type in ["PINN-GAT", "PINN-GIN", "PINN-MPNN", "PINN-Transformer"]:
    model_name = g_type.replace("PINN-", "")
    r2_list, mae_list = [], []
    for fold_idx, (poly, tr, te, p_sc, t_sc) in enumerate(folds_data):
        tr_loader = DataLoader(tr, batch_size=32, shuffle=True)
        te_loader = DataLoader(te, batch_size=32, shuffle=False)
        m = GenericPINN_GNN(gnn_type=model_name).to(device)
        r2, mae, _ = train_and_eval(
            m, tr_loader, te_loader, t_sc, p_sc, lambda_phys=0.1, epochs=60
        )
        r2_list.append(r2)
        mae_list.append(mae)
    mean_r2 = np.mean(r2_list)
    mean_mae = np.mean(mae_list)
    gnn_results.append(
        {"GNN_Architecture": g_type, "Mean_R2": mean_r2, "Mean_MAE_cp": mean_mae}
    )
    print(f"{g_type:<18} | Mean R2 = {mean_r2:.4f} | Mean MAE = {mean_mae:.2f} cp")

df_gnn = pd.DataFrame(gnn_results)
df_gnn.to_csv("results/benchmark_gnn_models.csv", index=False)
print("Saved: results/benchmark_gnn_models.csv\n")

print("=" * 80)
print("EXPERIMENT 2: Lambda Physics Weight Sensitivity Sweep (Fold 1)")
print("=" * 80)

poly, tr, te, p_sc, t_sc = folds_data[0]
tr_loader = DataLoader(tr, batch_size=32, shuffle=True)
te_loader = DataLoader(te, batch_size=32, shuffle=False)

lambda_results = []
for l_val in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    m = GenericPINN_GNN(gnn_type="GAT").to(device)
    r2, mae, viol = train_and_eval(
        m, tr_loader, te_loader, t_sc, p_sc, lambda_phys=l_val, epochs=60
    )
    lambda_results.append(
        {
            "Lambda": l_val,
            "Validation_R2": r2,
            "MAE_cp": mae,
            "Violation_Rate_Pct": viol,
        }
    )
    print(
        f"Lambda = {l_val:<4} | Val R2 = {r2:.4f} | MAE = {mae:.2f} cp | Violations = {viol:.2f}%"
    )

df_lambda = pd.DataFrame(lambda_results)
df_lambda.to_csv("results/benchmark_lambda_sweep.csv", index=False)
print("Saved: results/benchmark_lambda_sweep.csv\n")
