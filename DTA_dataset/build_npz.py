import os
import argparse
import numpy as np
import pandas as pd


def load_index(npz_path, key_field):
    """key -> row index 매핑만 만든다 (임베딩 자체는 복사하지 않고 원본 배열을 그대로 유지)."""
    data = np.load(npz_path, allow_pickle=True)
    keys = data[key_field]
    index_map = {str(k): i for i, k in enumerate(keys)}
    return index_map, data


def build_one(
    csv_path, out_path,
    protein_index, protein_data, drug_index, drug_data,
    protein_col, smiles_col, affinity_col, id_col,
    strict=True,
):
    """csv 1개 + 이미 로드된 임베딩 lookup으로 npz 1개를 만든다.

    strict=True면 매칭 0건일 때 예외를 던지고(단일 파일 CLI 실행용),
    strict=False면 경고만 찍고 None을 반환한다(여러 파일을 한 번에 처리하는 배치용).
    """
    usecols = {protein_col, smiles_col, affinity_col}
    if id_col:
        usecols.add(id_col)
    df = pd.read_csv(csv_path, usecols=list(usecols))

    protein_idx = df[protein_col].astype(str).map(protein_index)
    drug_idx = df[smiles_col].astype(str).map(drug_index)
    mask = protein_idx.notna() & drug_idx.notna()

    matched = df[mask]
    skipped = len(df) - len(matched)

    if len(matched) == 0:
        msg = (
            f"{csv_path}: 매칭되는 샘플이 하나도 없습니다. "
            f"--protein_col/--smiles_col이 임베딩 npz를 만들 때 쓴 id_col/seq_col과 같은 값을 쓰는지 확인하세요."
        )
        if strict:
            raise RuntimeError(msg)
        print(f"⚠️  {msg}")
        return None

    protein_idx = protein_idx[mask].to_numpy(dtype=np.int64)
    drug_idx = drug_idx[mask].to_numpy(dtype=np.int64)
    ids = matched[id_col].astype(str).to_numpy() if id_col else matched.index.astype(str).to_numpy()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        ID=ids,
        SMILES=matched[smiles_col].astype(str).to_numpy(),
        protein=matched[protein_col].astype(str).to_numpy(),
        affinity=matched[affinity_col].to_numpy(dtype=np.float32),
        drug_embedding=drug_data["drug_embedding"][drug_idx].astype(np.float32),
        protein_embedding=protein_data["mean"][protein_idx].astype(np.float32),
        protein_max_embedding=protein_data["max"][protein_idx].astype(np.float32),
    )
    print(f"✅ {csv_path} → {out_path} (samples={len(ids)}, skipped={skipped})")
    return len(ids), skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "split csv(SMILES/서열/affinity 포함) + 단백질/약물 임베딩 npz를 조인해 "
            "DTA_model/train.py, predict.py가 바로 읽을 수 있는 학습/예측용 npz를 만든다."
        )
    )
    parser.add_argument("--csv", required=True, help="변환할 split csv (예: train_data.csv)")
    parser.add_argument(
        "--protein_emb_npz", required=True,
        help="PLM/extract_protein_embeddings.py 출력 npz (protein, mean, max 키)",
    )
    parser.add_argument(
        "--drug_emb_npz", required=True,
        help="DLM/extract_drug_embeddings.py 출력 npz (SMILES, drug_embedding 키)",
    )
    parser.add_argument("--protein_col", default="target_accession", help="csv에서 단백질 식별자 컬럼 (protein_emb_npz의 키와 일치해야 함)")
    parser.add_argument("--smiles_col", default="SMILES", help="csv에서 SMILES 컬럼 (drug_emb_npz의 키와 일치해야 함)")
    parser.add_argument("--affinity_col", default="affinity")
    parser.add_argument("--id_col", default=None, help="샘플 ID로 쓸 컬럼 (없으면 row index 사용)")
    parser.add_argument("--out", required=True, help="출력 npz 경로")
    args = parser.parse_args()

    protein_index, protein_data = load_index(args.protein_emb_npz, "protein")
    drug_index, drug_data = load_index(args.drug_emb_npz, "SMILES")
    print(f"▶ protein lookup: {len(protein_index)}개, drug lookup: {len(drug_index)}개")

    build_one(
        args.csv, args.out,
        protein_index, protein_data, drug_index, drug_data,
        args.protein_col, args.smiles_col, args.affinity_col, args.id_col,
        strict=True,
    )


if __name__ == "__main__":
    main()

# 사용 예시 (ChEMBL IC50, fold1, train split)
# python DTA_dataset/build_npz.py \
#   --csv DTA_dataset/chembl_35/IC50/fold1/train_data.csv \
#   --protein_emb_npz emb/protein/chembl35_protein.npz \
#   --drug_emb_npz emb/drug/chembl35_drug.npz \
#   --protein_col target_accession --smiles_col SMILES --id_col compound_chembl_id \
#   --out DTA_model_npz/IC50/fold1/train_data.npz
#
# 여러 fold/split을 한 번에 처리하려면 preprocess_dataset.py를 쓰면 임베딩을
# 한 번만 로드해서 재사용하므로 훨씬 빠르다.
