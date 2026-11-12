"""Application Streamlit — Système de recommandation d'animes.

Point d'entrée : main.py (lance ce fichier avec Streamlit).
La logique métier (chargement des données + algorithmes) est dans backend.py.

Pages :
1. Exploration des données
2. Recommandations par animé (contenu TF-IDF + collaboratif kNN)
3. Recommandations pour un utilisateur (profil contenu + hybride + baseline popularité)
4. Nouvel utilisateur (genres cochés + notes données)
5. Filtrage collaboratif (RMSE user-based / item-based)
6. Évaluation des stratégies (precision@10, couverture)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import backend as be

st.set_page_config(
    page_title="SysRec Animes",
    page_icon="🎌",
    layout="wide",
)

st.title("🎌 Système de recommandation d'animes")
st.caption(
    "Filtrage basé sur le contenu (TF-IDF sur les genres) + filtrage collaboratif "
    "(kNN item-based, prédiction user/item) — dataset Kaggle *anime-recommendations-database*."
)


# ---------------------------------------------------------------------------
# Chargements en cache
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du dataset Kaggle et construction des modèles…")
def ressources():
    animes, ratings = be.load_data()
    rated = be.rated_only(ratings)          # sans les -1
    rec = be.AnimeRecommender(animes)       # TF-IDF contenu
    knn = be.ItemKNN(rated)                 # kNN collaboratif item-based
    popularite = be.popularity_counts(rated)
    return animes, ratings, rated, rec, knn, popularite


@st.cache_data(show_spinner=True)
def utilisateurs_actifs(rated: pd.DataFrame, min_likes: int = 5) -> pd.Series:
    """Utilisateurs ayant au moins `min_likes` animés aimés (note >= 8)."""
    aimes = rated[rated["rating"] >= be.LIKE_THRESHOLD]
    return aimes["user_id"].value_counts()[lambda s: s >= min_likes]


@st.cache_data(show_spinner=True)
def run_evaluation(sample_size: int, n_users: int, n_animes: int) -> pd.DataFrame:
    """Évaluation precision@k + couverture des 3 stratégies, avec cache par paramètres."""
    train, aimes_test = be.build_eval_split(rated, sample_size=sample_size)
    pop_train = be.popularity_counts(train)

    def contenu(u, notes):
        return rec.recommend_for_user(u, notes)["anime_id"].tolist()

    def popularite(u, notes):
        return be.recommend_popular(u, notes, pop_train)

    def hybride(u, notes):
        return rec.recommend_hybrid(u, notes, pop_train)["anime_id"].tolist()

    return be.evaluate_strategies(
        {"contenu": contenu, "popularité": popularite, "hybride": hybride},
        train,
        aimes_test,
    )


animes, ratings, rated, rec, knn, popularite = ressources()
n_ratings_rated = len(rated)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Exploration des données",
        "🎬 Recommandations par animé",
        "👤 Recommandations pour un utilisateur",
        "✨ Nouvel utilisateur",
        "👥 Filtrage collaboratif",
        "📈 Évaluation des stratégies",
    ],
)
st.sidebar.divider()
st.sidebar.metric("Animés", f"{animes['anime_id'].nunique():,}")
st.sidebar.metric("Utilisateurs", f"{rated['user_id'].nunique():,}")
st.sidebar.metric("Notes (hors -1)", f"{n_ratings_rated:,}")


def afficher_reco(df: pd.DataFrame, extra_cols: bool = True) -> None:
    """Affiche un tableau de recommandations proprement."""
    cols = ["name", "genre", "type", "episodes", "rating", "members"]
    if extra_cols and "score" in df.columns:
        cols.append("score")
    st.dataframe(
        df[cols],
        width="stretch",
        hide_index=True,
        column_config={
            "rating": st.column_config.NumberColumn("note moyenne", format="%.2f"),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
        },
    )


# ---------------------------------------------------------------------------
# 1. Exploration
# ---------------------------------------------------------------------------
if page.startswith("📊"):
    st.header("📊 Exploration des données")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Animés", f"{animes['anime_id'].nunique():,}")
    c2.metric("Utilisateurs", f"{rated['user_id'].nunique():,}")
    c3.metric("Notes (hors -1)", f"{n_ratings_rated:,}")
    sparsity = 100 * (1 - n_ratings_rated / (rated["user_id"].nunique() * animes["anime_id"].nunique()))
    c4.metric("Densité de la matrice", f"{100 - sparsity:.4f} %")

    st.subheader("Distribution des notes")
    st.bar_chart(rated["rating"].value_counts().sort_index(), x_label="note", y_label="nombre de notes")

    left, right = st.columns(2)
    with left:
        st.subheader("Répartition par type")
        st.bar_chart(animes["type"].value_counts())
    with right:
        st.subheader("Genres les plus fréquents")
        genres = animes["genre"].str.split(", ").explode()
        st.bar_chart(genres.value_counts().head(15))

    st.subheader("Top animés (moyenne bayésienne)")
    st.caption(
        "Moyenne pondérée par le nombre de notes (cf. Lab1_RS), pour éviter que des animés "
        "avec une seule note 10 n'écrasent le classement."
    )
    afficher_reco(be.bayesian_top(animes, rated, 20), extra_cols=False)

    with st.expander("Aperçu brut des données"):
        c1, c2 = st.columns(2)
        c1.write("**anime.csv**")
        c1.dataframe(animes.head(), width="stretch", hide_index=True)
        c2.write("**rating.csv**")
        c2.dataframe(ratings.head(), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 2. Recommandations par animé
# ---------------------------------------------------------------------------
elif page.startswith("🎬"):
    st.header("🎬 « Si vous avez aimé cet animé… »")
    st.caption(
        "À gauche : filtrage **basé sur le contenu** (similarité cosinus TF-IDF des genres). "
        "À droite : filtrage **collaboratif** (kNN sur la matrice utilitaire animés × utilisateurs)."
    )

    noms = sorted(rec.indices.index)
    choix = st.selectbox("Choisissez un animé", noms, index=noms.index("Death Note") if "Death Note" in noms else 0)
    k = st.slider("Nombre de recommandations", 5, 30, 10)

    anime_row = animes[animes["name"] == choix].iloc[0]
    st.info(f"**{choix}** — {anime_row['genre']} ({anime_row['type']}, note moyenne {anime_row['rating']})")

    gauche, droite = st.columns(2)
    with gauche:
        st.subheader("Contenu (TF-IDF)")
        afficher_reco(rec.similar_to_anime(choix, k))
    with droite:
        st.subheader("Collaboratif (kNN)")
        ids = knn.similar_animes(int(anime_row["anime_id"]), k)
        if ids:
            df_knn = animes.set_index("anime_id").loc[ids].reset_index()
            afficher_reco(df_knn, extra_cols=False)
        else:
            st.warning("Animé trop peu noté pour le kNN.")

# ---------------------------------------------------------------------------
# 3. Recommandations pour un utilisateur
# ---------------------------------------------------------------------------
elif page.startswith("👤"):
    st.header("👤 Recommandations personnalisées (contenu)")
    st.caption(
        "Profil de l'utilisateur = moyenne TF-IDF des animés qu'il a notés **8 ou plus** ; "
        "les animés qu'il a déjà notés sont exclus des résultats."
    )

    u_counts = utilisateurs_actifs(rated)
    user_id = st.selectbox(
        "Utilisateur (ayant au moins 5 animés aimés)",
        u_counts.index.sort_values(),
        format_func=lambda u: f"user {u}  ({u_counts[u]} animés aimés)",
    )

    notes_user = rated[rated["user_id"] == user_id]
    aimes = notes_user[notes_user["rating"] >= be.LIKE_THRESHOLD]
    nb_notes = st.slider("Nombre de recommandations", 5, 30, 10)

    gauche, droite = st.columns([1, 1])
    with gauche:
        st.subheader(f"Ses {len(aimes)} animés aimés")
        st.dataframe(
            animes[animes["anime_id"].isin(aimes["anime_id"])][["name", "genre", "rating"]],
            width="stretch",
            hide_index=True,
        )
    with droite:
        st.subheader("Son profil de genres (TF-IDF)")
        poids = rec.profile_terms(aimes["anime_id"], top=12)
        if not poids.empty:
            st.bar_chart(poids)

    st.divider()
    st.subheader(f"Top {nb_notes} recommandations")
    tab_contenu, tab_pop, tab_hybride = st.tabs(["🎯 Contenu", "📈 Popularité (baseline)", "🔀 Hybride"])
    with tab_contenu:
        afficher_reco(rec.recommend_for_user(user_id, rated, nb_notes))
    with tab_pop:
        ids = be.recommend_popular(user_id, rated, popularite, nb_notes)
        afficher_reco(animes[animes["anime_id"].isin(ids)].set_index("anime_id").loc[ids].reset_index(), extra_cols=False)
    with tab_hybride:
        afficher_reco(rec.recommend_hybrid(user_id, rated, popularite, nb_notes))

# ---------------------------------------------------------------------------
# 4. Nouvel utilisateur
# ---------------------------------------------------------------------------
elif page.startswith("✨"):
    st.header("✨ Recommandations pour un nouvel utilisateur")
    st.caption(
        "Profil construit à partir des genres cochés « à l'inscription » (comme dans le notebook), "
        "affiné par les animés que l'utilisateur note ici."
    )

    all_genres = sorted({g for ligne in animes["genre"] for g in ligne.split(", ") if g})
    cochés = st.multiselect("Genres préférés", all_genres, default=["Action", "Sci-Fi"])

    st.markdown("**Optionnel — notez quelques animés pour affiner le profil :**")
    noms = sorted(rec.indices.index)
    c1, c2, c3 = st.columns(3)
    choix_notes = {}
    for col, default in zip((c1, c2, c3), ("Death Note", "Steins;Gate", "Hunter x Hunter")):
        with col:
            nom = st.selectbox("Animé", noms, index=noms.index(default) if default in noms else 0, key=f"nom_{default}")
            choix_notes[nom] = st.slider("Note", 1, 10, 8, key=f"note_{default}")

    genres_str = " ".join(cochés)
    liked_ids = animes.loc[animes["name"].isin([n for n, note in choix_notes.items() if note >= be.LIKE_THRESHOLD]), "anime_id"].tolist()
    k = st.slider("Nombre de recommandations", 5, 30, 10)

    if not cochés and not liked_ids:
        st.warning("Choisissez au moins un genre ou notez un animé.")
    else:
        afficher_reco(rec.recommend_for_new_user(genres_str, liked_ids, k))

# ---------------------------------------------------------------------------
# 5. Filtrage collaboratif
# ---------------------------------------------------------------------------
elif page.startswith("👥"):
    st.header("👥 Filtrage collaboratif — prédiction de notes")
    st.caption(
        "Matrice users × animes construite sur les utilisateurs et animés les plus actifs, "
        "similarité = **cosinus ajusté** (centré par la moyenne de chaque utilisateur), "
        "prédiction centrée utilisateur. RMSE calculé sur les notes cachées du test."
    )

    sparsity = knn.X
    st.metric(
        "Sparsité de la matrice utilitaire",
        f"{100 * (1 - sparsity.count_nonzero() / (sparsity.shape[0] * sparsity.shape[1])):.2f} %",
    )

    c1, c2, c3 = st.columns(3)
    n_users = c1.number_input("Nb d'utilisateurs (top actifs)", 200, 4000, 1500, 100)
    n_animes = c2.number_input("Nb d'animés (top notés)", 200, 4000, 1500, 100)
    lancer = c3.checkbox("Lancer / relancer l'expérience", value=True)

    if lancer:
        with st.spinner("Calcul des corrélations et des prédictions…"):
            resultats = be.cf_experiment(rated, int(n_users), int(n_animes))
        st.bar_chart(resultats["RMSE"])
        st.dataframe(resultats, width="stretch")
        st.caption(
            "Interprétation : les deux approches CF battent nettement la baseline (moyenne globale), "
            "signe que la similarité entre utilisateurs/items porte une vraie information. "
            "L'item-based est légèrement devant ici ; sur un échantillon plus petit, le CF se dégrade "
            "vite à cause de la sparsité — d'où l'intérêt de l'approche contenu, qui ne dépend pas de "
            "l'historique des notes."
        )

# ---------------------------------------------------------------------------
# 6. Évaluation
# ---------------------------------------------------------------------------
elif page.startswith("📈"):
    st.header("📈 Comparaison des stratégies")
    st.caption(
        "precision@10 : part des 10 recommandations réellement aimées par l'utilisateur "
        "(notes >= 8 cachées dans le test). Couverture : nombre d'animés différents recommandés "
        "à l'ensemble des utilisateurs (diversité)."
    )

    c1, c2 = st.columns(2)
    taille = c1.select_slider("Taille de l'échantillon de notes", options=[10_000, 25_000, 50_000, 100_000], value=50_000)
    bouton = c2.button("🚀 Lancer l'évaluation", type="primary", width="stretch")

    if bouton or "eval_df" in st.session_state:
        with st.spinner("Évaluation des 3 stratégies (contenu, popularité, hybride)…"):
            df_eval = run_evaluation(taille, rec.matrix.shape[0], rec.matrix.shape[1])
            st.session_state["eval_df"] = df_eval

        st.dataframe(st.session_state["eval_df"], width="stretch")
        st.bar_chart(st.session_state["eval_df"]["réussite@10 (%)"])

        st.info(
            "**Lecture des résultats** : la baseline popularité a une precision souvent comparable, "
            "mais une couverture très faible (tout le monde reçoit à peu près la même liste). "
            "L'hybride contenu × popularité combine les deux : il remonte les animés de genre "
            "proche *et* suffisamment notés pour être fiables."
        )
