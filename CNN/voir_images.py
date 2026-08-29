"""Apercu des images d'une piste : python CNN/voir_images.py Monaco_dr"""
import glob
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

RACINE = Path(__file__).resolve().parent.parent
piste = sys.argv[1]
lignes, colonnes = 3, 4

fichier = sorted(glob.glob(str(RACINE / "data" / piste / "*.parquet")))[0]
table = pq.read_table(fichier)
pas = table.num_rows // (lignes * colonnes)

planche = Image.new("RGB", (160 * colonnes, 120 * lignes))
for n in range(lignes * colonnes):
    img = Image.open(io.BytesIO(table["image"][n * pas].as_py()))
    planche.paste(img, (160 * (n % colonnes), 120 * (n // colonnes)))

sortie = RACINE / "data" / f"apercu_{piste}.png"
planche.save(sortie)
print("ouvre :", sortie)
