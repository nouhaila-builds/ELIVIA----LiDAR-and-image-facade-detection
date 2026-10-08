"""
Extraction de l'archive nuScenes mini (v1.0-mini.tgz).

Pourquoi ce script ?
- Sur Windows, l'extraction dans un chemin long (ex: OneDrive\\...\\projet\\...) peut échouer (Long Path).
- Ici on conseille un chemin court (ex: C:\\nuscmini) puis on travaille avec ce root.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_tgz(archive_path: Path, out_dir: Path) -> None:
    archive_path = archive_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive introuvable: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as t:
        members = t.getmembers()
        # Extraction simple (progress léger)
        total = len(members)
        for i, m in enumerate(members, start=1):
            t.extract(m, path=out_dir)
            if i % 1000 == 0 or i == total:
                print(f"Extraction: {i}/{total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extraire v1.0-mini.tgz (nuScenes mini).")
    parser.add_argument("--archive", default="v1.0-mini.tgz", help="Chemin vers l'archive .tgz.")
    parser.add_argument(
        "--out",
        default=r"C:\nuscmini",
        help=r"Dossier de sortie (chemin court recommandé sous Windows).",
    )
    args = parser.parse_args()

    extract_tgz(Path(args.archive), Path(args.out))
    print("\nOK. Racine dataset:", Path(args.out).resolve())
    print("Doit contenir: samples/, sweeps/, v1.0-mini/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


