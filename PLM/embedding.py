import torch
import pandas as pd
import numpy as np

from model import ProteinMLM
from dataprocessing import (
    Bert_input_data,
    vocab,
    load_distance_dict_from_matrix_csv,
)

# =========================
# 설정
# =========================
csv_path = "/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/dataset/DAVIS-complete/unique_fasta.csv"
sequence_col = "FASTA"     # 임베딩 대상 서열
protein_col = "target_accession"              # protein ID

distance_csv_path = "six_aa_pairwise_distance.csv"
ckpt_path = "checkpoint_step500000.pt"

max_len = 1024
save_path = (
    "/HDD1/phj318/work/Chem_DTA/inception_PLM_DTA/int_sincos_embedding_V2/unique/DAVIS-complete_embeddings.npz"
)   

# =========================
# 디바이스
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 거리 사전 로드
# =========================
distance_dict = load_distance_dict_from_matrix_csv(distance_csv_path)

# =========================
# 모델 로딩
# =========================
model = ProteinMLM(
    vocab_size=len(vocab),
    embed_dim=1024,
    num_layers=16,
    num_heads=16,
    ffn_embed_dim=4096,
    padding_idx=vocab["[PAD]"],
    dropout=0.1,
)

ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# =========================
# CSV 로딩
# =========================
df = pd.read_csv(csv_path)

proteins = df[protein_col].tolist()
sequences = df[sequence_col].tolist()

print(f"▶ Total samples: {len(sequences)}")

# =========================
# CLS 기반 Mean Embedding 추출
# =========================
seq_list = []
mean_list = []
max_list = []
# min_list = []
protein_list = []

with torch.no_grad():
    for i, (protein, seq) in enumerate(zip(proteins, sequences)):
        data = Bert_input_data(
            seq=seq,
            vocab=vocab,
            distance_dict=distance_dict,
            max_len=max_len,
        )

        input_ids = data["input_ids"].unsqueeze(0).to(device)           # (1, T)
        position_ids = data["position_ids"].unsqueeze(0).to(device)    # (1, T)
        attention_mask = data["attention_mask"].unsqueeze(0).to(device)  # (1, T)

        # =========================
        # PLM forward
        # =========================
        _, token_representations = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        # (1, T, C) → (T, C)
        token_representations = token_representations.squeeze(0)

        # =========================
        # Mean Pooling (CLS / EOS 제거)
        # =========================
        tokens_len = int(attention_mask.sum().item())  # CLS~EOS 포함

        # CLS(0), EOS(tokens_len-1) 제거 후 padding (max_len-2 = 1022)
        actual_seq_emb = token_representations[1:tokens_len - 1]  # (L, C)
        
        # 고정 크기 (1022, 1024)로 Padding
        seq_emb = torch.zeros((max_len - 2, 1024), device=device)
        seq_emb[:actual_seq_emb.size(0)] = actual_seq_emb

        mean_rep = actual_seq_emb.mean(dim=0)  # (C,)
        max_rep  = actual_seq_emb.max(dim=0)[0] 
        # min_rep  = actual_seq_emb.min(dim=0)[0] 
        
        seq_list.append(seq_emb.cpu().numpy())
        mean_list.append(mean_rep.cpu().numpy())
        max_list.append(max_rep.cpu().numpy())
        # min_list.append(min_rep.cpu().numpy())
        protein_list.append(protein)

        if (i + 1) % 50 == 0 or (i + 1) == len(sequences):
            print(f"✓ Processed {i + 1}/{len(sequences)}")

# =========================
# NPZ 저장
# =========================
protein_array = np.array(protein_list)
seq_array = np.stack(seq_list, axis=0).astype(np.float32)  # (N, 1022, 1024)
mean_array = np.stack(mean_list, axis=0).astype(np.float32)  # (N, 1024)
max_array = np.stack(max_list, axis=0).astype(np.float32)  # (N, 1024)
# min_array = np.stack(min_list, axis=0).astype(np.float32)  # (N, 1024)

np.savez_compressed(
    save_path,
    protein=protein_array,
    seq=seq_array,
    mean=mean_array,
    max=max_array,
    # min=min_array,
)

print(
    f"\n✅ Saved embeddings to {save_path}\n"
    f"   - proteins: {protein_array.shape}\n"
    f"   - seq embedding: {seq_array.shape}\n"
    f"   - mean embedding: {mean_array.shape}\n"
    f"   - max embedding: {max_array.shape}\n"
    # f"   - min embedding: {min_array.shape}"    
)
