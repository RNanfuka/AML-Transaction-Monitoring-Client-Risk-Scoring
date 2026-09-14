from pathlib import Path
import shutil
import kagglehub

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
TARGET_DIR = Path("data/raw")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target = TARGET_DIR / "HI-Small_Trans.csv"
    if target.exists():
        print(f"Dataset already exists: {target}")
        return
    cache_dir = Path(kagglehub.dataset_download(DATASET, path="HI-Small_Trans.csv"))

    candidates = [cache_dir] if cache_dir.is_file() else list(cache_dir.rglob("HI-Small_Trans.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Downloaded dataset but HI-Small_Trans.csv was not found. "
            f"Inspect: {cache_dir}"
        )

    source = candidates[0]
    target = TARGET_DIR / "HI-Small_Trans.csv"
    shutil.copy2(source, target)
    print(f"Copied dataset to {target}")
    print("The raw dataset is intentionally git-ignored.")


if __name__ == "__main__":
    main()
