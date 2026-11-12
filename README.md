# **Mini-projet : Mise en place d’un système de recommandation**

Le but de ce mini-projet est de mettre en place un système de recommandation qui se base sur lesalgorithmes de filtrage collaboratif et de filtrage basé sur le contenu.

Le dataset choisi est [ici](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).

## 🚀 Lancer l'application Streamlit

```bash
uv sync                # installe les dépendances (dont streamlit)
uv run python main.py  # ouvre http://localhost:8501
```

Options Streamlit supplémentaires après `--` (ex. `uv run python main.py -- --server.port 8888`).
Au premier lancement, le dataset est téléchargé via `kagglehub` puis mis en cache local.

## 🗂️ Structure du code

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée : lance `app.py` via l'API Streamlit |
| `app.py` | Interface Streamlit (6 pages) et mise en cache des données/modèles |
| `backend.py` | Système de recommandation : chargement/nettoyage des données, TF-IDF contenu, kNN collaboratif, prédiction user/item, évaluation |

## 📄 Pages de l'application

1. **Exploration des données** — volumes, densité, distributions, top animés (moyenne bayésienne)
2. **Recommandations par animé** — « si vous avez aimé X » : contenu (TF-IDF + cosinus) vs collaboratif (kNN)
3. **Recommandations pour un utilisateur** — profil TF-IDF de ses animés notés ≥ 8, recommandations contenu / popularité / hybride
4. **Nouvel utilisateur** — profil à partir des genres cochés (+ notes optionnelles)
5. **Filtrage collaboratif** — RMSE des prédictions user-based et item-based (cosinus ajusté) vs baseline
6. **Évaluation des stratégies** — precision@10, réussite@10 et couverture du catalogue

## 📓 Notebooks

- `projet.ipynb` : découverte du dataset
- `partie2_content_eval.ipynb` : contenu (TF-IDF), évaluation, hybride, CF
- `examples/` : labs du cours (MovieLens, ML) dont s'inspirent les méthodes

