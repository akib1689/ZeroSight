#!/usr/bin/env python3
"""Download all benchmark datasets.

Run once:
    uv run python scripts/download_datasets.py

Downloads to data/:
    data/hotpotqa/      — HF Dataset (Arrow format)
    data/musique/       — HF Dataset (Arrow format)
    data/fever/         — HF Dataset (Arrow format)
    data/2wiki/         — HF Dataset (Arrow format)
    data/spider/        — Spider 1.0 (JSON + SQLite databases)
    data/test_db/       — Employees MySQL sample DB (datacharmer/test_db v1.0.7)
    data/hitab/         — HiTab hierarchical-table QA (microsoft/HiTab, commit-pinned)

Note: BIRD Mini-Dev is not downloaded by this script — it must be obtained
separately and placed at data/bird/minidev/MINIDEV/.

After downloading, dataset adapters use load_from_disk() or read files
directly — no network at runtime.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_hotpotqa():
    from datasets import load_dataset

    print("Downloading HotpotQA...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds.save_to_disk(DATA_DIR / "hotpotqa")
    print(f"  Saved {len(ds)} items to data/hotpotqa/")


def download_musique():
    from datasets import load_dataset

    print("Downloading MuSiQue...")
    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    ds.save_to_disk(DATA_DIR / "musique")
    print(f"  Saved {len(ds)} items to data/musique/")


def download_fever():
    from datasets import load_dataset

    print("Downloading FEVER (copenlu/fever_gold_evidence)...")
    ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    ds.save_to_disk(DATA_DIR / "fever")
    print(f"  Saved {len(ds)} items to data/fever/")


def download_2wiki():
    from datasets import load_dataset

    print("Downloading 2WikiMultihopQA...")
    ds = load_dataset("voidful/2WikiMultihopQA", split="validation")
    ds.save_to_disk(DATA_DIR / "2wiki")
    print(f"  Saved {len(ds)} items to data/2wiki/")


def download_spider():
    """Download Spider 1.0 (SQLite, dev split) from HuggingFace.

    Saves dev.json + tables.json + database/ to data/spider/.
    """
    from datasets import load_dataset

    import json
    import zipfile

    print("Downloading Spider 1.0...")
    spider_dir = DATA_DIR / "spider"
    spider_dir.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace dataset first
    try:
        ds = load_dataset("xlangai/spider", split="validation")
        # Save as JSON
        items = []
        for item in ds:
            items.append(item)
        with open(spider_dir / "dev.json", "w") as f:
            json.dump(items, f, indent=2)
        print(f"  Saved {len(items)} items to data/spider/dev.json")
    except Exception:
        print("  HuggingFace dataset unavailable, trying direct download...")
        import urllib.request

        # Spider 1.0 from Yale (Google Drive mirror)
        url = "https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2W5gZp-s2ioA0k0LqQkDca"
        zip_path = spider_dir / "spider.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(spider_dir)
            zip_path.unlink()
            print(f"  Extracted to data/spider/")
        except Exception as e:
            print(
                f"  WARNING: Could not download Spider automatically: {e}\n"
                f"  Please download manually from "
                f"https://yale-lily.github.io/spider and extract to data/spider/"
            )
            return

    # Download databases if not present
    db_dir = spider_dir / "database"
    if not db_dir.exists():
        import urllib.request

        print("  Downloading Spider databases...")
        db_url = (
            "https://drive.google.com/uc?export=download&id"
            "=1iRDVHLr4mX2W5gZp-s2ioA0k0LqQkDca"
        )
        try:
            zip_path = spider_dir / "spider_db.zip"
            urllib.request.urlretrieve(db_url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(spider_dir)
            zip_path.unlink()
        except Exception as e:
            print(
                f"  WARNING: Could not download Spider databases: {e}\n"
                f"  Please download manually."
            )


TEST_DB_URL = "https://github.com/datacharmer/test_db/archive/refs/tags/v1.0.7.tar.gz"
TEST_DB_SHA256 = "dddfdb13228d9eec9fd76a3e316037a279768155e015f08fb5302bf6a6541250"


def _combine_test_db(src_dir: Path, out_path: Path) -> None:
    """Flatten test_db into one SQL script the docker entrypoint can pipe.

    - Inlines the `source <file> ;` directives (client-side, unsupported when
      piped without a cwd) — the .dump files are plain INSERTs.
    - Drops `source show_elapsed.sql` (info_schema timing query) and the
      binlog `flush ... logs` line (the mysql docker image disables binlog).
    """
    import re

    parts: list[str] = []
    for line in (src_dir / "employees.sql").read_text().splitlines(keepends=True):
        m = re.match(r"\s*source\s+(\S+)\s*;", line)
        if m:
            fname = m.group(1)
            if fname == "show_elapsed.sql":
                continue
            parts.append((src_dir / fname).read_text())
        elif re.match(r"\s*flush\s+/\*!50503 binary \*/\s*logs\s*;", line):
            continue
        else:
            parts.append(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))


def download_test_db():
    """Download datacharmer/test_db v1.0.7 (Employees MySQL sample DB).

    Checksum-verified, extracted to data/test_db/, and flattened into
    data/init/01-employees.sql which docker-compose feeds to the mysql
    container's /docker-entrypoint-initdb.d on first start
    (run `docker compose up -d`).
    """
    import hashlib
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    dest = DATA_DIR / "test_db"
    combined = DATA_DIR / "init" / "01-employees.sql"
    if (dest / "employees.sql").exists() and combined.exists():
        print("test_db already present at data/test_db/ — skipping.")
        return

    if not (dest / "employees.sql").exists():
        print("Downloading Employees sample DB (datacharmer/test_db v1.0.7)...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "test_db.tar.gz"
            urllib.request.urlretrieve(TEST_DB_URL, archive)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != TEST_DB_SHA256:
                raise RuntimeError(
                    f"test_db checksum mismatch: got {digest}, expected {TEST_DB_SHA256}"
                )
            with tarfile.open(archive) as tf:
                tf.extractall(tmp_path, filter="data")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(tmp_path / "test_db-1.0.7"), str(dest))
        print("  Saved to data/test_db/")

    print("  Generating data/init/01-employees.sql (combined init script)...")
    _combine_test_db(dest, combined)
    print(f"  Done ({combined.stat().st_size / 1e6:.0f} MB).")


HITAB_COMMIT = "d179602662b490249baf068a76fbe4137029126e"  # microsoft/HiTab HEAD 2025-12-16
HITAB_FILES = {
    "train_samples.jsonl": "25a3d58025bbf69322b2caa0b28a565e9c2db6aa61a1541d97c5d3f2ecc1d0fd",
    "dev_samples.jsonl": "567530ce270422d1f1bde767a732f5cb44ef2c717eacdfe43cb3ca6f990c2475",
    "test_samples.jsonl": "cfb8296a23d4481757e1e9f63874085c90bc498386aa0b522c9ef3df65c1f339",
    "test_samples_qualitycheck.jsonl": "989ae7be8d949bef659644c713d5db61275dae09c7198819978fa02cb3251165",
    "tables.zip": "c26f72f4a318c2a1a2c177f17d7cbc27db9e0bb4d30f0a7f53c037b0a15770b5",
    "LICENSE": "c536073ff6d5f6317f7eef4163aa7cf01f7a5b0f0a7a38f4a5fd187c605bd38b",
}


def download_hitab():
    """Download microsoft/HiTab (hierarchical-table QA, ACL 2022, C-UDA v1.0).

    Fetches train/dev/test jsonl (both test variants), all tables, and the
    license from raw.githubusercontent at a pinned commit, verifies sha256,
    and extracts tables to data/hitab/tables/. Both test variants are kept:
    the adapter's default `subset` variant (benchmarks/hitab/subset.json) is
    drawn from test_samples_qualitycheck.jsonl (Oct 2025 label corrections)
    and the original test_samples.jsonl remains available for literature
    comparability.
    """
    import hashlib
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    dest = DATA_DIR / "hitab"
    marker = dest / "tables"
    if marker.exists() and (dest / "test_samples_qualitycheck.jsonl").exists():
        print("HiTab already present at data/hitab/ — skipping.")
        return

    print("Downloading HiTab (microsoft/HiTab, C-UDA v1.0)...")
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/microsoft/HiTab/{HITAB_COMMIT}/data"
    lic_url = (
        f"https://raw.githubusercontent.com/microsoft/HiTab/{HITAB_COMMIT}/LICENSE"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fname, expected in HITAB_FILES.items():
            url = lic_url if fname == "LICENSE" else f"{base}/{fname}"
            fpath = tmp_path / fname
            urllib.request.urlretrieve(url, fpath)
            digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(
                    f"HiTab checksum mismatch for {fname}: got {digest}, "
                    f"expected {expected}"
                )
            print(f"  {fname}: OK ({fpath.stat().st_size / 1e6:.1f} MB)")

        # tables.zip extracts to tables/<source>/<table_id>.json — flatten the
        # doubled root so the final layout is data/hitab/tables/<source>/<id>.json
        with zipfile.ZipFile(tmp_path / "tables.zip") as zf:
            zf.extractall(tmp_path / "unzipped")
        inner = tmp_path / "unzipped" / "tables"
        if not inner.is_dir():
            raise RuntimeError("Unexpected tables.zip layout: no 'tables/' root")
        if marker.exists():
            shutil.rmtree(marker)
        shutil.move(str(inner), str(marker))
        for fname in HITAB_FILES:
            if fname != "tables.zip":
                (tmp_path / fname).replace(dest / fname)

    n_tables = sum(1 for _ in marker.glob("*/*.json"))
    print(f"  Saved to data/hitab/ ({n_tables} tables)")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # HuggingFace datasets
    for fn in [
        download_hotpotqa,
        download_musique,
        download_fever,
        download_2wiki,
        download_spider,
        download_test_db,
        download_hitab,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  WARNING: {fn.__name__} failed: {e}")

    print("\nDone. Data saved under data/. Run adapters use local files.")
    print("Note: BIRD Mini-Dev must be downloaded separately to data/bird/minidev/MINIDEV/")


if __name__ == "__main__":
    main()
