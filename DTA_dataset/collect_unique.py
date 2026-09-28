import argparse
import glob
import pandas as pd


def resolve_paths(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "여러 csv에서 (id_col, value_col) 쌍을 모아 중복 제거된 조회용 csv를 만든다. "
            "예: DAVIS fold1~5의 train/val/test csv에서 (protein, target_sequence)를 모아 "
            "unique_fasta.csv를, (drug_name, compound_iso_smiles)를 모아 unique_smiles.csv를 만든다."
        )
    )
    parser.add_argument(
        "--csv_glob", required=True, nargs="+",
        help='입력 csv 경로 또는 glob 패턴 (여러 개 가능). 쉘 확장을 막으려면 따옴표로 감쌀 것, 예: "DTA_dataset/DAVIS/fold*/*.csv"',
    )
    parser.add_argument("--id_col", required=True)
    parser.add_argument("--value_col", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = resolve_paths(args.csv_glob)
    if not paths:
        raise FileNotFoundError(f"입력 csv를 찾지 못했습니다: {args.csv_glob}")

    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=[args.id_col, args.value_col])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=[args.id_col]).reset_index(drop=True)

    merged.to_csv(args.out, index=False)
    print(
        f"✅ {len(paths)}개 csv에서 {before}행 → unique {args.id_col} {len(merged)}개 → {args.out}"
    )


if __name__ == "__main__":
    main()

# 사용 예시
# python DTA_dataset/collect_unique.py \
#   --csv_glob "DTA_dataset/DAVIS/fold*/*.csv" \
#   --id_col protein --value_col target_sequence \
#   --out DTA_dataset/DAVIS/unique_fasta.csv
