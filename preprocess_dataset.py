import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "DTA_dataset"))
from build_npz import load_index, build_one  # noqa: E402


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# DTA_model/train.py, predict.py가 fold 디렉토리마다 요구하는 파일명.
# test_non5 계열은 만들지 않는다 (파일이 커서 시간/용량을 많이 잡아먹고, train.py도
# 없으면 그냥 건너뛰도록 이미 패치돼 있음)
CHEMBL_SPLITS = {
    name: name
    for name in [
        "train_data", "valid_data", "test_data", "test_wild_data", "test_mutant_data",
    ]
}
# DAVIS는 val_data.csv로 존재하지만 train.py는 valid_data.npz를 찾고, test_non5 계열은 없음
DAVIS_SPLITS = {
    "train_data": "train_data",
    "val_data": "valid_data",
    "test_data": "test_data",
    "test_wild_data": "test_wild_data",
    "test_mutant_data": "test_mutant_data",
}

# emb/<그룹명>/{protein,drug}.npz 형태로 저장한다.
# DAVIS-complete/IC50/Kd/Ki 4개 폴더로 DTA_model/result, DTA_model_npz와 이름을 맞춘다.
DATASETS = {
    "DAVIS": dict(
        group="DAVIS-complete",
        fold_glob=_p("DTA_dataset", "DAVIS", "fold*", "*.csv"),
        protein_id_col="protein", protein_seq_col="target_sequence",
        drug_id_col="drug_name", drug_smiles_col="compound_iso_smiles",
        unique_fasta=_p("DTA_dataset", "DAVIS", "unique_fasta.csv"),
        unique_smiles=_p("DTA_dataset", "DAVIS", "unique_smiles.csv"),
        fold_dir_tmpl=_p("DTA_dataset", "DAVIS", "fold{i}"),
        out_dir_tmpl=_p("DTA_model_npz", "DAVIS-complete", "fold{i}"),
        splits=DAVIS_SPLITS,
    ),
}
for _name in ("IC50", "Kd", "Ki"):
    DATASETS[_name] = dict(
        group=_name,
        # unique 목록은 IC50+Kd+Ki가 겹치는 단백질/약물이 많아 chembl_35 전체에서 함께 모은다
        # (임베딩 자체는 4-1절처럼 데이터셋별 emb/<이름>/ 폴더에 각각 저장된다)
        fold_glob=_p("DTA_dataset", "chembl_35", "*", "fold*", "*.csv"),
        protein_id_col="target_accession", protein_seq_col="FASTA",
        drug_id_col="compound_chembl_id", drug_smiles_col="SMILES",
        unique_fasta=_p("DTA_dataset", "chembl_35", "unique_fasta.csv"),
        unique_smiles=_p("DTA_dataset", "chembl_35", "unique_smiles.csv"),
        fold_dir_tmpl=_p("DTA_dataset", "chembl_35", _name, "fold{i}"),
        out_dir_tmpl=_p("DTA_model_npz", _name, "fold{i}"),
        splits=CHEMBL_SPLITS,
    )

for _cfg in DATASETS.values():
    _cfg["protein_emb"] = _p("emb", _cfg["group"], "protein.npz")
    _cfg["drug_emb"] = _p("emb", _cfg["group"], "drug.npz")


def ensure_unique_lists(cfg, force):
    if force or not os.path.isfile(cfg["unique_fasta"]):
        run([
            sys.executable, _p("DTA_dataset", "collect_unique.py"),
            "--csv_glob", cfg["fold_glob"],
            "--id_col", cfg["protein_id_col"], "--value_col", cfg["protein_seq_col"],
            "--out", cfg["unique_fasta"],
        ])
    else:
        print(f"✓ {cfg['unique_fasta']} 이미 있음, 건너뜀")

    if force or not os.path.isfile(cfg["unique_smiles"]):
        run([
            sys.executable, _p("DTA_dataset", "collect_unique.py"),
            "--csv_glob", cfg["fold_glob"],
            "--id_col", cfg["drug_id_col"], "--value_col", cfg["drug_smiles_col"],
            "--out", cfg["unique_smiles"],
        ])
    else:
        print(f"✓ {cfg['unique_smiles']} 이미 있음, 건너뜀")


def ensure_protein_embedding(cfg, ckpt, force):
    if force or not os.path.isfile(cfg["protein_emb"]):
        os.makedirs(os.path.dirname(cfg["protein_emb"]), exist_ok=True)
        run([
            sys.executable, _p("PLM", "extract_protein_embeddings.py"),
            "--csv", cfg["unique_fasta"],
            "--id_col", cfg["protein_id_col"], "--seq_col", cfg["protein_seq_col"],
            "--ckpt", ckpt,
            "--save_path", cfg["protein_emb"],
        ])
    else:
        print(f"✓ {cfg['protein_emb']} 이미 있음, 건너뜀")


def ensure_drug_embedding(cfg, batch_size, force):
    if force or not os.path.isfile(cfg["drug_emb"]):
        os.makedirs(os.path.dirname(cfg["drug_emb"]), exist_ok=True)
        run([
            sys.executable, _p("DLM", "extract_drug_embeddings.py"),
            "--csv", cfg["unique_smiles"],
            "--smiles_col", cfg["drug_smiles_col"], "--id_col", cfg["drug_id_col"],
            "--batch_size", str(batch_size),
            "--save_path", cfg["drug_emb"],
        ])
    else:
        print(f"✓ {cfg['drug_emb']} 이미 있음, 건너뜀")


def build_all_npz(cfg, force):
    # 임베딩은 fold/split이 몇 개든 딱 한 번만 로드해서 재사용한다
    # (매번 새로 로드하면 chembl35_drug.npz 322MB를 파일마다 다시 읽어야 해서 느려짐)
    print(f"임베딩 로드 중: {cfg['protein_emb']}, {cfg['drug_emb']}")
    protein_index, protein_data = load_index(cfg["protein_emb"], "protein")
    drug_index, drug_data = load_index(cfg["drug_emb"], "SMILES")

    for i in range(1, 6):
        fold_in = cfg["fold_dir_tmpl"].format(i=i)
        fold_out = cfg["out_dir_tmpl"].format(i=i)
        os.makedirs(fold_out, exist_ok=True)

        for src_name, out_name in cfg["splits"].items():
            csv_path = os.path.join(fold_in, f"{src_name}.csv")
            out_path = os.path.join(fold_out, f"{out_name}.npz")

            if not os.path.isfile(csv_path):
                print(f"⚠️  {csv_path} 없음, 건너뜀")
                continue
            if not force and os.path.isfile(out_path):
                print(f"✓ {out_path} 이미 있음, 건너뜀")
                continue

            build_one(
                csv_path, out_path,
                protein_index, protein_data, drug_index, drug_data,
                cfg["protein_id_col"], cfg["drug_smiles_col"], "affinity", cfg["drug_id_col"],
                strict=False,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "데이터셋 이름만 주면: unique 단백질/약물 목록 생성 → PLM/ChemBERTa 임베딩 추출 → "
            "fold별 학습용 npz 조립까지 한 번에 처리한다. 이미 있는 결과물은 건너뛰고, "
            "--force를 주면 전부 다시 만든다."
        )
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--ckpt", default=_p("PLM", "checkpoint_step500000.pt"), help="PLM 체크포인트 경로")
    parser.add_argument("--batch_size", type=int, default=256, help="ChemBERTa 임베딩 추출 배치 크기")
    parser.add_argument("--force", action="store_true", help="이미 있는 결과도 다시 만든다")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    print(f"\n=== [{args.dataset}] 1/4 unique 단백질/약물 목록 ===")
    ensure_unique_lists(cfg, args.force)

    print(f"\n=== [{args.dataset}] 2/4 단백질 임베딩 ===")
    ensure_protein_embedding(cfg, args.ckpt, args.force)

    print(f"\n=== [{args.dataset}] 3/4 약물 임베딩 ===")
    ensure_drug_embedding(cfg, args.batch_size, args.force)

    print(f"\n=== [{args.dataset}] 4/4 fold별 학습용 npz 조립 ===")
    build_all_npz(cfg, args.force)

    print(f"\n✅ [{args.dataset}] 전처리 완료 → {cfg['out_dir_tmpl'].format(i='1..5')}")


if __name__ == "__main__":
    main()

# 사용 예시
# python preprocess_dataset.py --dataset DAVIS
# python preprocess_dataset.py --dataset IC50
# python preprocess_dataset.py --dataset Kd --force   # 전부 다시 생성
