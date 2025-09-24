"""Utilitat per importar fitxers CSV de trànsit a una base de dades SQLite."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REQUIRED_COLUMNS: Sequence[str] = ("idTram", "data", "estatActual", "estatPrevist")


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def _ensure_csv_folder(csv_folder: Path) -> Path:
    csv_folder = csv_folder.expanduser()
    if not csv_folder.exists() or not csv_folder.is_dir():
        raise FileNotFoundError(
            f"No s'ha trobat la carpeta de fitxers CSV: {csv_folder}"
        )
    return csv_folder


def _iter_csv_files(csv_folder: Path) -> Iterable[Path]:
    return sorted(csv_folder.glob("*.csv"))


def build_database(csv_folder: Path, db_path: Path, *, verbose: bool = False) -> int:
    """Construeix la base de dades SQLite amb les dades dels CSV.

    Parameters
    ----------
    csv_folder:
        Carpeta que conté els fitxers CSV.
    db_path:
        Fitxer de sortida de la base de dades SQLite.
    verbose:
        Si és ``True`` s'imprimeixen missatges de progrés.

    Returns
    -------
    int
        Nombre de registres inserits a la taula ``estat_transit``.
    """

    csv_folder = _ensure_csv_folder(Path(csv_folder))
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    inserted_rows = 0

    _log("Carregant fitxers...", verbose)
    csv_files = list(_iter_csv_files(csv_folder))
    _log(f"S'han trobat {len(csv_files)} fitxers CSV.", verbose)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS estat_transit (
                idTram INTEGER,
                data INTEGER,
                estatActual INTEGER,
                estatPrevist INTEGER,
                source_file TEXT
            )
            """
        )
        conn.commit()

        for file in csv_files:
            try:
                df = pd.read_csv(file, on_bad_lines="skip")
            except Exception as exc:  # pragma: no cover - dependència externa
                _log(f"❌ Error carregant {file.name}: {exc}", verbose)
                continue

            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"El fitxer {file.name} no conté les columnes requerides: {missing_columns}"
                )

            df = df[list(REQUIRED_COLUMNS)].copy()
            df["source_file"] = file.name

            df.to_sql("estat_transit", conn, if_exists="append", index=False)
            inserted_rows += len(df)
            _log(f"✔ Fitxer carregat: {file.name}", verbose)

        conn.commit()
    finally:
        conn.close()

    return inserted_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construeix la base de dades de trànsit a partir de fitxers CSV"
    )
    parser.add_argument(
        "--csv-folder",
        type=Path,
        default=Path("data") / "estat_transit",
        help="Carpeta on es troben els fitxers CSV.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data") / "transit.db",
        help="Ruta on es crearà el fitxer SQLite.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra missatges de progrés durant el procés de càrrega.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inserted = build_database(args.csv_folder, args.db_path, verbose=args.verbose)
    except FileNotFoundError as exc:  # pragma: no cover - integració CLI
        print(f"❌ {exc}")
        return 1

    print(
        "✅ Base de dades creada correctament. "
        f"S'han inserit {inserted} registres a {args.db_path}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - punt d'entrada CLI
    raise SystemExit(main())
