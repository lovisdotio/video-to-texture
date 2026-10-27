"""Upload the >100 MiB HQ dataset with fal's multipart file repository.

Small-file upload helpers often reject files above 100 MiB;
the lossless 1024 px dataset is larger.  This helper validates the ZIP, uploads it
with the supported multipart API, and writes the URL consumed by ``fit_run_hq.py``.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from fal.toolkit import File


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "dataset_v03_hq_masked.zip"
URL_FILE = ROOT / "dataset_url_hq.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--url-file", type=Path, default=URL_FILE)
    args = parser.parse_args()
    archive = args.archive.resolve()
    url_file = args.url_file.resolve()
    if not archive.is_file() or archive.stat().st_size < 1024:
        raise SystemExit(f"Missing or empty HQ dataset archive: {archive}")
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        if not {"manifest.json", "dataset.npz"}.issubset(names):
            raise SystemExit("Dataset ZIP lacks root manifest.json or dataset.npz")
        if handle.testzip() is not None:
            raise SystemExit("Dataset ZIP CRC validation failed")
    uploaded = File.from_path(
        archive,
        content_type="application/zip",
        multipart=True,
    )
    if not uploaded.url.startswith(("http://", "https://")):
        raise RuntimeError(f"fal returned an invalid dataset URL: {uploaded.url!r}")
    url_file.parent.mkdir(parents=True, exist_ok=True)
    url_file.write_text(uploaded.url + "\n")
    print(f"DATASET_URL_FILE_READY {url_file}")
    print(uploaded.url)


if __name__ == "__main__":
    main()
