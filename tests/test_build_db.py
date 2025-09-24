import sqlite3
import tempfile
import unittest
from pathlib import Path

from build_db import build_database


class BuildDatabaseTests(unittest.TestCase):
    def test_build_database_inserts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_dir = tmp_path / "csv"
            csv_dir.mkdir()

            (csv_dir / "202401.csv").write_text(
                "idTram,data,estatActual,estatPrevist\n1,20240101,2,3\n",
                encoding="utf-8",
            )
            (csv_dir / "202402.csv").write_text(
                "idTram,data,estatActual,estatPrevist\n2,20240201,1,1\n",
                encoding="utf-8",
            )

            db_path = tmp_path / "transit.db"
            inserted = build_database(csv_dir, db_path)

            self.assertEqual(inserted, 2)
            self.assertTrue(db_path.exists())

            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute(
                    "SELECT idTram, data, estatActual, estatPrevist, source_file "
                    "FROM estat_transit ORDER BY data"
                ).fetchall()
            finally:
                conn.close()

            self.assertEqual(
                rows,
                [
                    (1, 20240101, 2, 3, "202401.csv"),
                    (2, 20240201, 1, 1, "202402.csv"),
                ],
            )

    def test_missing_csv_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_folder = tmp_path / "not_there"
            db_path = tmp_path / "transit.db"

            with self.assertRaises(FileNotFoundError):
                build_database(missing_folder, db_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
