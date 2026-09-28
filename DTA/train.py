import os
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Integration
from process_data import DTAData
from train_and_test import train

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------
# Seed utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Global config
# ----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
TEST_BATCH_SIZE = 512
MAX_SMILES_LEN = 128
max_len= 1024

# ----------------------------
# One fold run
# ----------------------------
def run_fold(
    fold,
    seed,
    data_dir,
    result_dir,
    # pe_type,
    # max_len,
    resume=False
):
    # =========================
    # Data path
    # =========================
    fold_dir = os.path.join(data_dir, f"fold{fold+1}")

    # =========================
    # Result save path
    # DAVIS_result/int_sincos/len1024/fold1
    # =========================
    save_path = os.path.join(
        result_dir,
        # pe_type,
        # f"len{max_len}",
        f"fold{fold+1}"
    )
    os.makedirs(save_path, exist_ok=True)

    MODEL_NAME = os.path.join(save_path)

    # =========================
    # Logging
    # =========================
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=f"{MODEL_NAME}.log",
        level=logging.DEBUG
    )

    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs"))

    # =========================
    # Load data
    # =========================
    df_train = np.load(os.path.join(fold_dir, "train_data.npz"))
    df_val   = np.load(os.path.join(fold_dir, "valid_data.npz"))
    df_test  = np.load(os.path.join(fold_dir, "test_data.npz"))

    df_test_wild = np.load(os.path.join(fold_dir, "test_wild_data.npz"))
    df_test_mutant = np.load(os.path.join(fold_dir, "test_mutant_data.npz"))
    df_test_non5 = np.load(os.path.join(fold_dir, "test_non5_data.npz"))
    df_test_wild_non5 = np.load(os.path.join(fold_dir, "test_wild_non5_data.npz"))
    df_test_mutant_non5 = np.load(os.path.join(fold_dir, "test_mutant_non5_data.npz"))

    # =========================
    # Dataset
    # =========================
    train_set = DTAData(
        df_train["drug_embedding"], df_train["protein_embedding"],df_train["protein_max_embedding"], df_train["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    val_set = DTAData(
        df_val["drug_embedding"], df_val["protein_embedding"],df_val["protein_max_embedding"], df_val["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_set = DTAData(
        df_test["drug_embedding"], df_test["protein_embedding"],df_test["protein_max_embedding"], df_test["affinity"],
        device, MAX_SMILES_LEN, max_len
    )

    test_wild_set = DTAData(
        df_test_wild["drug_embedding"], df_test_wild["protein_embedding"],df_test_wild["protein_max_embedding"], df_test_wild["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_mutant_set = DTAData(
        df_test_mutant["drug_embedding"], df_test_mutant["protein_embedding"],df_test_mutant["protein_max_embedding"], df_test_mutant["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_non5_set = DTAData(
        df_test_non5["drug_embedding"], df_test_non5["protein_embedding"],df_test_non5["protein_max_embedding"], df_test_non5["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_wild_non5_set = DTAData(
        df_test_wild_non5["drug_embedding"], df_test_wild_non5["protein_embedding"],df_test_wild_non5["protein_max_embedding"], df_test_wild_non5["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_mutant_non5_set = DTAData(
        df_test_mutant_non5["drug_embedding"], df_test_mutant_non5["protein_embedding"],df_test_mutant_non5["protein_max_embedding"], df_test_mutant_non5["affinity"],
        device, MAX_SMILES_LEN, max_len
    )

    # =========================
    # DataLoader
    # =========================
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )
    valid_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )

    test_wild_loader = DataLoader(test_wild_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                  generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    test_mutant_loader = DataLoader(test_mutant_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                    generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    test_non5_loader = DataLoader(test_non5_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                  generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    test_wild_non5_loader = DataLoader(test_wild_non5_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                       generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    test_mutant_non5_loader = DataLoader(test_mutant_non5_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                         generator=g, num_workers=8, pin_memory=True, persistent_workers=True)

    # =========================
    # Model
    # =========================
    import torch.nn as nn
    model = Integration(
        drug_model=nn.Identity(),
        protein_model=nn.Identity(),
        drug_out_dim=384,
        protein_dim=1024,
        proj_dim=1024,
        hidden_sizes=[1024, 512, 256, 1],
        dropout=0.1,
    ).to(device)

    # =========================
    # Train
    # =========================
    train(
        model,
        train_loader, valid_loader,
        test_loader,
        test_wild_loader, test_mutant_loader,
        test_non5_loader, test_wild_non5_loader, test_mutant_non5_loader,
        writer,
        MODEL_NAME,
        save_path,
        lr=1e-4,
        epoch=500,
        resume=resume
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Embedding dataset dir (contains fold1~5)")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Root dir to save results")

    # parser.add_argument("--pe_type", type=str, required=True,
    #                     help="e.g. int_sincos / float_sincos")
    #parser.add_argument("--max_len", type=int, choices=[1024, 2048], required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    for fold in range(5):
        print(f"\n🚀 Starting Fold {fold+1}")
        run_fold(
            fold,
            args.seed,
            args.data_dir,
            args.result_dir,
            #args.pe_type,
            #args.max_len,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
# python PLM_ChemBERT_77_DTA_V2/train.py \
#   --data_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/DAVIS-complete \
#   --result_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/PLM_ChemBERT_77_DTA_V2/result/DAVIS-complete \
#   --resume && \
# python PLM_ChemBERT_77_DTA_V2/train.py \
#   --data_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/Kd \
#   --result_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/PLM_ChemBERT_77_DTA_V2/result/Kd \
#   --resume && \
# python PLM_ChemBERT_77_DTA_V2/train.py \
#   --data_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/Ki \
#   --result_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/PLM_ChemBERT_77_DTA_V2/result/Ki \
#   --resume && \
# python PLM_ChemBERT_77_DTA_V2/train.py \
#   --data_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/protein_embedding/max_len_1024/int_sincos_77/IC50 \
#   --result_dir /HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/PLM_ChemBERT_77_DTA_V2/result/IC50 \
#   --resume