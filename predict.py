import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from src.MolProcessing import PreProcess, Molecule, get_mol_max_length, get_motif_max_length

def read_inputs(csv_path, npy_path):
    df = pd.read_csv(csv_path)
    smiles_list = df.iloc[:, 0].tolist()
    if df.shape[1] > 1:
        labels = df.iloc[:, 1].values.astype(np.float32)
    else:
        labels = None

    features = np.load(npy_path)
    assert len(features) == len(smiles_list), "npy特征数量与SMILES数量不一致！"
    smiles_to_features = {s: features[i] for i, s in enumerate(smiles_list)}
    return df, smiles_list, labels, smiles_to_features

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', required=True, type=str)
    parser.add_argument('--smiles', required=True, type=str)
    parser.add_argument('--transformer', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def batch_predict(model, mol_list, smiles_to_features, batch_size, device, mean=None, std=None):
    model.eval()
    preds = []

    for i in range(0, len(mol_list), batch_size):
        mol_batch = mol_list[i:i + batch_size]
        inputs, _ = PreProcess(mol_batch, smiles_to_features)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(inputs, output_feature=False)
            if mean is not None and std is not None:
                output = output * std + mean  # ⚠️ 使用预测数据集的 mean/std 做反标准化
            preds.append(output.detach().cpu())

    return torch.cat(preds, dim=0).numpy()

def main():
    args = parse_args()

    dataset = args.property
    input_csv = args.smiles
    input_npy = args.transformer
    output_csv = args.output
    NUM_FOLDS = args.num_folds
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, smiles_list, labels, smiles_to_features = read_inputs(input_csv, input_npy)

    # 获取 max_len 和 max_motif
    all_smiles_y = [(smi, [0]) for smi in smiles_list]
    max_len = get_mol_max_length(all_smiles_y)
    max_motif = get_motif_max_length(all_smiles_y)

    mol_list = []
    for smi in tqdm(smiles_list, desc='Processing molecules'):
        mol = Molecule([smi, [0]], dataset, seed=0, bool_random=False,
                       max_len=max_len, max_motif=max_motif)
        mol_list.append(mol if mol.exist else None)

    valid_idx = [i for i, m in enumerate(mol_list) if m is not None]
    valid_mols = [mol_list[i] for i in valid_idx]

    # 计算预测数据集的 mean/std 用于回归反标准化
    if labels is not None:
        mean = torch.tensor(np.mean(labels), dtype=torch.float32).to(device)
        std = torch.tensor(np.std(labels), dtype=torch.float32).to(device)
        print(f"⚠️ 使用预测数据集的 mean/std 进行反标准化：mean={mean.item():.4f}, std={std.item():.4f}")
    else:
        mean = std = None

    all_preds = []
    for fold in range(NUM_FOLDS):
        model_path = f"./data/{dataset}/fold_{fold}/model.pth"
        print(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
        model.to(device)

        preds = batch_predict(model, valid_mols, smiles_to_features, BATCH_SIZE, device, mean, std)
        preds = preds.reshape(len(valid_mols), -1).squeeze()
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=1)  # shape: (N_valid, NUM_FOLDS)

    # 写入到 df
    for i in range(NUM_FOLDS):
        col = f'fold_{i}'
        pred_column = np.full(len(df), np.nan)
        for idx, v in zip(valid_idx, all_preds[:, i]):
            pred_column[idx] = v
        df[col] = pred_column

    # 如果有标签，计算 RMSE
    if labels is not None:
        rmses = []
        summary_row = {col: "" for col in df.columns}

        for i in range(NUM_FOLDS):
            pred = df[f'fold_{i}'].values
            mask = ~np.isnan(pred)
            rmse = np.sqrt(np.mean((pred[mask] - labels[mask]) ** 2))
            rmses.append(rmse)
            summary_row[f"fold_{i}"] = f"{rmse:.3f}"

        # 添加一列：写入 Mean ± Std
        if "RMSE" not in df.columns:
            df["RMSE"] = ""
        summary_row["RMSE"] = f"{np.mean(rmses):.3f} +/- {np.std(rmses):.3f}"


        # 添加这一行
        df.loc[len(df)] = summary_row
        df.to_csv(output_csv, index=False)

        print(f"\n✅ Prediction results saved to: {output_csv}")
        print("Per-fold RMSE:", np.round(rmses, 4))
        print("Mean ± Std RMSE: {:.4f} ± {:.4f}".format(np.mean(rmses), np.std(rmses)))

if __name__ == '__main__':
    main()
