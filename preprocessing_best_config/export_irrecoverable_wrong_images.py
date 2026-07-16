from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "testing" / "irrecoverable_wrong_images_8pipelines"

PREDICTION_FILES = [
    ROOT / "outputs/testing/preprocessing_course_benchmark/predictions_test_train_baseline.csv",
    ROOT / "outputs/testing/ml_official_preprocessing_benchmark/predictions_test_raw_rgb.csv",
    ROOT / "outputs/testing/preprocessing_course_benchmark/predictions_test_clahe_clip1_tile4.csv",
    ROOT / "outputs/testing/preprocessing_course_benchmark/predictions_test_homomorphic_filter.csv",
    ROOT / "outputs/testing/preprocessing_combinations_benchmark/predictions_test_clahe_rl_deblur_bilateral.csv",
    ROOT / "outputs/testing/preprocessing_combinations_benchmark/predictions_test_rl_deblur_bilateral_lowpass.csv",
    ROOT / "outputs/testing/ml_official_preprocessing_benchmark/predictions_test_zero_dce.csv",
    ROOT / "outputs/testing/ml_official_preprocessing_benchmark/predictions_test_restormer_motion_deblur_native.csv",
]


def method_name(path: Path) -> str:
    return path.stem.replace("predictions_test_", "")


def build_irrecoverable_table() -> pd.DataFrame:
    by_image: dict[str, dict] = {}

    for prediction_file in PREDICTION_FILES:
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)

        method = method_name(prediction_file)
        frame = pd.read_csv(prediction_file)
        for row in frame.to_dict("records"):
            image_path = str(row["image_path"])
            record = by_image.setdefault(
                image_path,
                {
                    "image_path": image_path,
                    "file": Path(image_path).name,
                    "target": str(row["target"]),
                    "source_name": row.get("source_name", ""),
                    "plate_type": row.get("plate_type", ""),
                    "correct_count": 0,
                    "best_edit_distance": 10**9,
                    "best_prediction": "",
                    "best_method": "",
                    "train_baseline_prediction": "",
                },
            )

            if bool(row["exact"]):
                record["correct_count"] += 1

            edit_distance = int(row["edit_distance"])
            if edit_distance < record["best_edit_distance"]:
                record["best_edit_distance"] = edit_distance
                record["best_prediction"] = str(row["prediction"])
                record["best_method"] = method

            if method == "train_baseline":
                record["train_baseline_prediction"] = str(row["prediction"])

    rows = [record for record in by_image.values() if record["correct_count"] == 0]
    rows.sort(key=lambda item: (item["source_name"], item["plate_type"], item["file"]))
    return pd.DataFrame(rows)


def copy_images(table: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied_paths = []

    for index, row in table.reset_index(drop=True).iterrows():
        source = Path(row["image_path"])
        suffix = source.suffix.lower()
        copied_name = (
            f"{index + 1:02d}_target_{row['target']}_pred_{row['best_prediction']}"
            f"_err_{row['best_edit_distance']}{suffix}"
        )
        destination = OUTPUT_DIR / copied_name
        shutil.copy2(source, destination)
        copied_paths.append(str(destination))

    table = table.copy()
    table["copied_image_path"] = copied_paths
    return table


def main() -> None:
    table = build_irrecoverable_table()
    table = copy_images(table)

    csv_path = OUTPUT_DIR / "irrecoverable_wrong_images_8pipelines.csv"
    columns = [
        "file",
        "target",
        "train_baseline_prediction",
        "best_prediction",
        "best_edit_distance",
        "best_method",
        "source_name",
        "plate_type",
        "image_path",
        "copied_image_path",
    ]
    table.to_csv(csv_path, columns=columns, index=False, encoding="utf-8-sig")

    print(f"Exported {len(table)} images")
    print(csv_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
