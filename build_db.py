import sqlite3
import pandas as pd
from pathlib import Path

# Ruta a la carpeta de fitxers CSV
csv_folder = Path("D:\TFG\Data\Open Data BCN\Trànsit\Estat_transit")

# Ruta del fitxer SQLite
db_path = Path("D:/TFG/Data/Open Data BCN/transit.db")

# Connexió a SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Crear taula si no existeix
cursor.execute('''
CREATE TABLE IF NOT EXISTS estat_transit (
    idTram INTEGER,
    data INTEGER,
    estatActual INTEGER,
    estatPrevist INTEGER,
    source_file TEXT
)
''')
conn.commit()

# Processar tots els CSV
csv_files = sorted(csv_folder.glob("*.csv"))
print(f"Carregant {len(csv_files)} fitxers...")

for file in csv_files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        df["source_file"] = file.name

        # Assegura't que només tens les columnes correctes
        df = df[["idTram", "data", "estatActual", "estatPrevist", "source_file"]]

        # Escriure a la base de dades
        df.to_sql("estat_transit", conn, if_exists="append", index=False)
        print(f"✔ Fitxer carregat: {file.name}")
    except Exception as e:
        print(f"❌ Error carregant {file.name}: {e}")

conn.close()
print("✅ Base de dades creada correctament.")
