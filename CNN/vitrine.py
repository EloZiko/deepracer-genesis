"""Mesure, choisit deux pistes, et fabrique les videos comparatives.

Le but : montrer ce que coute la perception apprise. Meme politique, meme piste,
meme camera -- seule la source des 7 valeurs change (simulateur exact vs CNN).

    caffeinate -di ../.venv/bin/python vitrine.py politique_reference.pt

Deroulement, ~40 min :
  1. evalue chaque piste dans les deux cas          (1 sous-processus par scene)
  2. choisit DEPUIS LES CHIFFRES la piste ou le CNN aide le plus, et celle ou
     les deux sont les plus proches -- pas de piste choisie a la main
  3. tourne les 4 videos de flotte et les colle deux a deux, cote a cote
  4. ecrit runs/vitrine.json

Une scene camera par sous-processus : le contexte OpenGL de pyrender est global
au processus, et detruire une scene precedente invalide celle qui tourne.
"""

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
SORTIE = RACINE / "runs" / "vitrine"

PISTES = ("2022_march_open", "arctic_open", "jyllandsringen_open",
          "hamption_pro", "thunder_hill_pro", "Tokyo_Training_track",
          "dubai_open", "Monaco")
CAS = ("parfait", "cnn")
POLICE = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def bandeau(texte, largeur, vers):
    """Fabrique un bandeau de titre en PNG.

    On ne passe pas par le filtre `drawtext` de ffmpeg : il n'est pas compile
    dans toutes les builds (absent de celle de Homebrew ici), et son absence ne
    se voit qu'a l'execution.
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype(POLICE, 30)
    except OSError:
        f = ImageFont.load_default()
    im = Image.new("RGBA", (largeur, 62), (30, 33, 40, 225))
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), texte, font=f)
    d.text(((largeur - (r - l)) / 2 - l, (62 - (b - t)) / 2 - t), texte,
           font=f, fill=(255, 255, 255, 255))
    im.save(vers)
    return vers


def evaluer(ckpt, piste, cas):
    p = subprocess.run([sys.executable, "comparer_cnn.py", ckpt, "--une", piste, cas],
                       cwd=ICI, capture_output=True, text=True)
    l = next((x for x in p.stdout.splitlines() if x.startswith("RESULTAT ")), None)
    if l is None:
        print("      echec :", "\n".join((p.stdout + p.stderr).splitlines()[-4:]))
    return json.loads(l[9:]) if l else None


def filmer(ckpt, piste, cas):
    """Rend la video de flotte ; retourne son chemin, ou None."""
    p = subprocess.run([sys.executable, "video_flotte.py", ckpt, piste, cas],
                       cwd=ICI, capture_output=True, text=True)
    f = RACINE / "runs" / "videos" / cas / f"flotte_{piste}.mp4"
    if not f.exists():
        print("      echec video :", "\n".join((p.stdout + p.stderr).splitlines()[-4:]))
        return None
    return f


def coller(gauche, droite, sortie):
    """Colle deux videos cote a cote, chacune coiffee de son bandeau."""
    import subprocess as sp
    largeur = int(sp.run(["ffprobe", "-v", "error", "-select_streams", "v",
                          "-show_entries", "stream=width", "-of", "csv=p=0",
                          str(gauche)], capture_output=True, text=True).stdout.strip())
    bg = bandeau("PERCEPTION PARFAITE", largeur, SORTIE / "_bandeau_g.png")
    bd = bandeau("VIA LE CNN", largeur, SORTIE / "_bandeau_d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(gauche), "-i", str(droite),
         "-i", str(bg), "-i", str(bd), "-filter_complex",
         "[0:v][2:v]overlay=0:0[g];[1:v][3:v]overlay=0:0[d];[g][d]hstack=inputs=2[v]",
         "-map", "[v]", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
         str(sortie)], check=True)
    return sortie


def choisir(res):
    """La piste ou le CNN aide le plus, et celle ou les deux se ressemblent le plus."""
    ecart = {p: r["parfait"]["offtrack_rate"] - r["cnn"]["offtrack_rate"]
             for p, r in res.items()}
    aide = max(ecart, key=ecart.get)
    if ecart[aide] <= 0.02:      # aucune piste ou le CNN reduit les sorties
        aide = max(res, key=lambda p: res[p]["cnn"]["mean_progress_m"]
                   - res[p]["parfait"]["mean_progress_m"])
    proche = min((p for p in res if p != aide),
                 key=lambda p: abs(ecart[p]) + abs(res[p]["cnn"]["mean_progress_m"]
                 - res[p]["parfait"]["mean_progress_m"]) / 50)
    return aide, proche


def tableau(res):
    print(f"\n{'':24} {'hors piste':>21} {'progres (m)':>21}")
    print(f"{'piste':24} {'parfait':>10} {'cnn':>10} {'parfait':>10} {'cnn':>10}")
    for p, r in res.items():
        print(f"{p:24} {r['parfait']['offtrack_rate']:10.2f} "
              f"{r['cnn']['offtrack_rate']:10.2f} "
              f"{r['parfait']['mean_progress_m']:10.1f} "
              f"{r['cnn']['mean_progress_m']:10.1f}")
    n = len(res)
    m = lambda c, k: sum(r[c][k] for r in res.values()) / n
    print(f"{'MOYENNE':24} {m('parfait','offtrack_rate'):10.2f} "
          f"{m('cnn','offtrack_rate'):10.2f} {m('parfait','mean_progress_m'):10.1f} "
          f"{m('cnn','mean_progress_m'):10.1f}", flush=True)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "politique_reference.pt"
    SORTIE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n1/3  mesures sur {len(PISTES)} pistes, deux sources de perception\n")
    res = {}
    for i, piste in enumerate(PISTES, 1):
        r = {}
        for cas in CAS:
            print(f"  {i:2}/{len(PISTES)}  {piste:24} {cas:8}", flush=True)
            m = evaluer(ckpt, piste, cas)
            if m is None:
                break
            r[cas] = m
        if len(r) == 2:
            res[piste] = r
            (SORTIE / "vitrine.json").write_text(json.dumps(res, indent=1))
    if len(res) < 2:
        print("pas assez de pistes mesurees, on s'arrete")
        return
    tableau(res)

    aide, proche = choisir(res)
    print(f"\n2/3  pistes retenues, d'apres les chiffres")
    print(f"  le CNN aide le plus  : {aide}")
    print(f"  les deux se valent   : {proche}\n", flush=True)

    print("3/3  videos (4 rendus + 2 collages)\n")
    videos = {}
    for etiq, piste in (("aide", aide), ("pareil", proche)):
        cotes = [filmer(ckpt, piste, cas) for cas in CAS]
        if None in cotes:
            continue
        f = coller(cotes[0], cotes[1], SORTIE / f"{etiq}_{piste}.mp4")
        videos[etiq] = {"piste": piste, "fichier": str(f)}
        print(f"  {f}", flush=True)

    (SORTIE / "vitrine.json").write_text(json.dumps(
        {"mesures": res, "videos": videos, "checkpoint": ckpt}, indent=1))
    print(f"\ntermine en {(time.time()-t0)/60:.0f} min")
    print(f"tout est dans {SORTIE}")


if __name__ == "__main__":
    main()
