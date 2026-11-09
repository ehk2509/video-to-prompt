# video-to-prompt

Transforme une vidéo (fichier ou URL) en prompts prêts à coller dans un générateur vidéo, pour créer des vidéos similaires.

L'outil découpe la vidéo en plans, analyse des images de chaque plan avec un modèle de vision, transcrit la parole, puis écrit les prompts au format du générateur choisi. Tout tourne en local par défaut.

## Installation

Il faut Python 3.10 ou plus, ffmpeg dans le PATH et [Ollama](https://ollama.com).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
ollama pull qwen3-vl:8b-instruct
```

## Utilisation

```powershell
.\.venv\Scripts\video2prompt.exe ma_video.mp4
.\.venv\Scripts\video2prompt.exe "https://exemple.com/video" -t ltx,wan,generic --variations 5
```

| Option | Rôle | Défaut |
| --- | --- | --- |
| `-t`, `--target` | Générateurs visés : `ltx`, `wan`, `generic` | `ltx,wan` |
| `--variations` | Nombre de nouvelles idées dans le même style | `3` |
| `-b`, `--backend` | `ollama` (local) ou `claude` (API) | `ollama` |
| `-m`, `--model` | Modèle à utiliser | selon le backend |
| `-o`, `--output` | Dossier de sortie | `output` |
| `--max-shots` | Plans analysés au maximum, les plus longs d'abord | `24` |
| `--frames-per-shot` | Images analysées par plan | `4` |
| `--scene-threshold` | Sensibilité aux coupes, de 0 à 1 (plus bas = plus de plans) | `0.3` |
| `--no-transcribe` | Ignore la parole | désactivé |
| `--whisper-model` | Modèle Whisper (`small`, `medium`, `large-v3`...) | `small` |

## Résultat

Chaque vidéo produit un dossier `output/<nom-de-la-video>/` :

- `prompts.md` : le style global puis, pour chaque générateur, un prompt complet, un modèle réutilisable avec `[SUBJECT]`, `[SETTING]`, `[ACTION]`, un prompt négatif, des variations et un prompt par plan.
- `result.json` : les mêmes données et l'analyse détaillée, pour les réutiliser dans un autre programme.
- `frames/` : les images extraites. Celle indiquée pour chaque plan peut servir de départ en image-vers-vidéo.

## Utiliser Claude à la place d'Ollama

Définis `ANTHROPIC_API_KEY`, puis ajoute `-b claude`. Le modèle par défaut est `claude-opus-5-5`. Cette option est payante à l'usage.

## À savoir

- Le modèle de vision regarde des images fixes, il déduit donc le mouvement en les comparant. Les mouvements de caméra subtils peuvent lui échapper.
- Seule la parole est réellement écoutée. La musique n'est pas analysée et les ambiances sonores sont déduites de l'image.
- Les sous-titres incrustés et filigranes sont volontairement exclus des prompts, car les générateurs les déforment. Ils restent consultables dans `result.json`.
- Utilise `qwen3-vl:8b-instruct` plutôt que `qwen3-vl:8b`. Ce dernier raisonne longuement et renvoie parfois une réponse vide.
- Vérifie que tu as le droit de télécharger et de réutiliser les vidéos que tu analyses.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
