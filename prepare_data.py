from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset


BASE_URL = "https://huggingface.co/datasets/bbaaaa/iwslt14-de-en-preprocess/resolve/refs%2Fconvert%2Fparquet/de-en"



def save_split(split_name: str, data: Dataset, output_dir: Path) -> None:
    de_file = output_dir / f"{split_name}.de"
    en_file = output_dir / f"{split_name}.en"

    with open(de_file, "w", encoding="utf-8") as f_de, open(en_file, "w", encoding="utf-8") as f_en:
        for row in data:
            row_dict = cast(dict[str, Any], row)
            translation = cast(dict[str, str], row_dict["translation"])
            f_de.write(translation["de"] + "\n")
            f_en.write(translation["en"] + "\n")


if __name__ == "__main__":
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{BASE_URL}/train/0000.parquet",
            "validation": f"{BASE_URL}/validation/0000.parquet",
            "test": f"{BASE_URL}/test/0000.parquet",
        },
    )

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    save_split("train", dataset["train"], output_dir)
    save_split("val", dataset["validation"], output_dir)
    save_split("test", dataset["test"], output_dir)