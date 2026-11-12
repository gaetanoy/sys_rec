"""Backend du système de recommandation d'animes.
"""

from __future__ import annotations

import html
from math import sqrt

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DATASET = "CooperUnion/anime-recommendations-database"
SEED = 42
LIKE_THRESHOLD = 8  # note >= 8 <=> animé « aimé » (choix du notebook partie2)


# ---------------------------------------------------------------------------
# 1. Chargement et nettoyage des données
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge anime.csv et rating.csv via kagglehub (cache local) et nettoie."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    animes_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS, DATASET, "anime.csv"
    )
    ratings_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS, DATASET, "rating.csv"
    )

    animes = animes_df[
        ["anime_id", "name", "genre", "type", "episodes", "rating", "members"]
    ].copy()
    ratings = ratings_df[["user_id", "anime_id", "rating"]].copy()

    # Les noms contiennent des entités HTML (&#039; etc.)
    animes["name"] = animes["name"].astype(str).apply(html.unescape)
    animes["genre"] = animes["genre"].fillna("").astype(str)
    animes["type"] = animes["type"].fillna("Unknown")
    animes["episodes"] = pd.to_numeric(animes["episodes"], errors="coerce")

    return animes, ratings


def rated_only(ratings: pd.DataFrame) -> pd.DataFrame:
    """Retire les notes -1 (animés listés mais non notés)."""
    return ratings[ratings["rating"] != -1].copy()


def bayesian_top(animes: pd.DataFrame, rated: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top animés par moyenne bayésienne (cf. Lab1_RS : moyenne pondérée par le nombre de notes)."""
    stats = rated.groupby("anime_id")["rating"].agg(["count", "mean"])
    C, m = stats["count"].mean(), stats["mean"].mean()
    stats["bayesian_avg"] = (C * m + stats["mean"] * stats["count"]) / (C + stats["count"])
    stats = stats.merge(animes[["anime_id", "name", "genre", "type", "episodes", "rating", "members"]], on="anime_id")
    return stats.sort_values("bayesian_avg", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# 2. Filtrage basé sur le contenu (TF-IDF sur les genres)
# ---------------------------------------------------------------------------

class AnimeRecommender:
    """Système basé sur le contenu : TF-IDF des genres + cosinus."""

    def __init__(self, animes: pd.DataFrame):
        self.animes = animes.reset_index(drop=True)
        self.anime_ids = self.animes["anime_id"]

        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.animes["genre"])

        # nom d'animé -> position dans la matrice (les doublons de nom sont dédupliqués)
        self.indices = (
            pd.Series(self.animes.index, index=self.animes["name"])
            .groupby(level=0)
            .first()
        )
        # anime_id -> position
        self.pos_of = pd.Series(self.animes.index, index=self.anime_ids)

    # ---- position helpers -------------------------------------------------

    def _positions(self, anime_ids: pd.Series) -> np.ndarray:
        """Positions (lignes de la matrice TF-IDF) des animés donnés."""
        return anime_ids.map(self.pos_of).dropna().astype(int).to_numpy()

    # ---- recommandation par animé -----------------------------------------

    def similar_to_anime(self, name: str, n: int = 10) -> pd.DataFrame:
        """Les n animés les plus similaires (genres) à celui dont le nom est donné."""
        if name not in self.indices.index:
            raise KeyError(f"Animé inconnu : {name!r}")
        idx = int(self.indices[name])
        sim = linear_kernel(self.matrix[idx], self.matrix).ravel()
        return self._top_from_scores(sim, n, exclude_idx=idx)

    # ---- profil utilisateur ------------------------------------------------

    def profile_terms(self, anime_ids: pd.Series, top: int = 10) -> pd.Series:
        """Termes du profil TF-IDF (genres aimés) d'un utilisateur, du plus important au moins important."""
        lignes = self._positions(anime_ids)
        if len(lignes) == 0:
            return pd.Series(dtype=float)
        profil = np.asarray(self.matrix[lignes].mean(axis=0)).reshape(1, -1)
        termes = self.vectorizer.get_feature_names_out()
        poids = pd.Series(profil.ravel(), index=termes)
        return poids.sort_values(ascending=False).head(top)

    def recommend_for_user(self, user_id: int, ratings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Recommandations contenu pour un utilisateur : profil = moyenne TF-IDF de ses animés aimés."""
        notes_user = ratings[ratings["user_id"] == user_id]
        aimes = notes_user.loc[notes_user["rating"] >= LIKE_THRESHOLD, "anime_id"]
        lignes = self._positions(aimes)
        if len(lignes) == 0:
            return self.animes.iloc[0:0].copy()  # DataFrame vide mais avec les colonnes

        profil = np.asarray(self.matrix[lignes].mean(axis=0)).reshape(1, -1)
        scores = linear_kernel(profil, self.matrix).ravel()
        # exclusion des animés déjà vus (notés, aimés ou pas)
        deja_vus = self.anime_ids.isin(notes_user["anime_id"]).to_numpy()
        scores[deja_vus] = -1
        return self._top_from_scores(scores, n)

    # ---- nouvel utilisateur ------------------------------------------------

    def recommend_for_new_user(
        self,
        genres: str,
        liked_ids: list[int] | None = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """Recommandations à partir des genres cochés à l'inscription (+ animés déjà aimés si fournis)."""
        profil = self.vectorizer.transform([genres])
        if liked_ids:
            lignes = self._positions(pd.Series(liked_ids))
            if len(lignes) > 0:
                profil_u = np.asarray(self.matrix[lignes].mean(axis=0)).reshape(1, -1)
                profil = np.vstack([profil.toarray(), profil_u]).mean(axis=0).reshape(1, -1)
        scores = linear_kernel(profil, self.matrix).ravel()
        scores[self._positions(pd.Series(liked_ids or []))] = -1
        return self._top_from_scores(scores, n)

    # ---- hybride -----------------------------------------------------------

    def recommend_hybrid(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        popularity: pd.Series,
        n: int = 10,
    ) -> pd.DataFrame:
        """Hybride : score contenu multiplié par log(1 + nb de notes de l'animé)."""
        notes_user = ratings[ratings["user_id"] == user_id]
        aimes = notes_user.loc[notes_user["rating"] >= LIKE_THRESHOLD, "anime_id"]
        lignes = self._positions(aimes)
        if len(lignes) == 0:
            return self.animes.iloc[0:0].copy()

        nb_notes = self.anime_ids.map(popularity).fillna(0).to_numpy()
        profil = np.asarray(self.matrix[lignes].mean(axis=0)).reshape(1, -1)
        scores = linear_kernel(profil, self.matrix).ravel() * np.log1p(nb_notes)
        deja_vus = self.anime_ids.isin(notes_user["anime_id"]).to_numpy()
        scores[deja_vus] = -1
        return self._top_from_scores(scores, n)

    # ---- interne -----------------------------------------------------------

    def _top_from_scores(self, scores: np.ndarray, n: int, exclude_idx: int | None = None) -> pd.DataFrame:
        if exclude_idx is not None:
            scores = scores.copy()
            scores[exclude_idx] = -1
        positions = np.argsort(-scores, kind="stable")[:n]
        out = self.animes.iloc[positions].copy()
        out["score"] = scores[positions].round(4)
        return out


# ---------------------------------------------------------------------------
# 3. Baseline popularité
# ---------------------------------------------------------------------------

def popularity_counts(ratings: pd.DataFrame) -> pd.Series:
    """Nombre de notes par anime_id, décroissant (index = anime_id)."""
    return ratings["anime_id"].value_counts()


def recommend_popular(user_id: int, ratings: pd.DataFrame, popularite: pd.Series, n: int = 10) -> list[int]:
    """Les n animés les plus notés que l'utilisateur n'a pas encore notés."""
    deja_vus = set(ratings.loc[ratings["user_id"] == user_id, "anime_id"])
    return [m for m in popularite.index if m not in deja_vus][:n]


# ---------------------------------------------------------------------------
# 4. Filtrage collaboratif
# ---------------------------------------------------------------------------

def build_utility_matrix(ratings: pd.DataFrame) -> tuple[csr_matrix, dict, dict]:
    """Matrice utilitaire sparse items x users (cf. Lab1_RS, create_X)."""
    N = ratings["user_id"].nunique()
    M = ratings["anime_id"].nunique()

    user_mapper = dict(zip(np.unique(ratings["user_id"]), range(N)))
    anime_mapper = dict(zip(np.unique(ratings["anime_id"]), range(M)))
    anime_inv_mapper = dict(zip(range(M), np.unique(ratings["anime_id"])))

    rows = [anime_mapper[a] for a in ratings["anime_id"]]
    cols = [user_mapper[u] for u in ratings["user_id"]]

    X = csr_matrix((ratings["rating"].astype(float), (rows, cols)), shape=(M, N))
    return X, anime_inv_mapper, anime_mapper


class ItemKNN:
    """kNN item-based : « parce que vous avez regardé X » (cf. Lab1_RS, find_similar_movies)."""

    def __init__(self, ratings: pd.DataFrame, metric: str = "cosine"):
        self.X, self.inv_mapper, self.mapper = build_utility_matrix(ratings)
        self.knn = NearestNeighbors(
            n_neighbors=11, algorithm="brute", metric=metric
        ).fit(self.X)

    def similar_animes(self, anime_id: int, k: int = 10) -> list[int]:
        if anime_id not in self.mapper:
            return []
        vec = self.X[self.mapper[anime_id]]
        neighbors = self.knn.kneighbors(vec.reshape(1, -1), n_neighbors=k + 1, return_distance=False).ravel()
        ids = [self.inv_mapper[i] for i in neighbors if self.inv_mapper[i] != anime_id]
        return ids[:k]


def _cosine(X: np.ndarray) -> np.ndarray:
    """Matrice de similarité cosinus entre les lignes de X."""
    dots = X @ X.T
    norms = np.sqrt((X**2).sum(axis=1))
    norms[norms == 0] = 1
    return dots / norms[:, np.newaxis] / norms[np.newaxis, :]


def predict_user_based(train_M: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    """Prédiction user-based : moyenne utilisateur + somme pondérée des écarts à la moyenne.

    Contrairement à la version du notebook (qui sommait sur toutes les cases, donc aussi
    sur les notes absentes remplies de 0), on ne somme que sur les notes réellement présentes.
    """
    mask = (train_M > 0).astype(float)
    mean_user = np.divide(
        train_M.sum(axis=1), mask.sum(axis=1),
        out=np.zeros(len(train_M)), where=mask.sum(axis=1) > 0,
    )
    adj = np.where(train_M > 0, train_M - mean_user[:, np.newaxis], 0.0)
    denom = np.abs(similarity).dot(mask)
    denom[denom == 0] = 1
    pred = mean_user[:, np.newaxis] + similarity.dot(adj) / denom
    return np.clip(pred, 1, 10)


def predict_item_based(train_M: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    """Prédiction item-based : moyenne de l'utilisateur ajustée par les items similaires qu'il a notés."""
    mask = (train_M > 0).astype(float)
    mean_user = np.divide(
        train_M.sum(axis=1), mask.sum(axis=1),
        out=np.zeros(len(train_M)), where=mask.sum(axis=1) > 0,
    )
    adj = np.where(train_M > 0, train_M - mean_user[:, np.newaxis], 0.0)
    sim_pos = np.clip(similarity, 0, None)  # seuls les items proches positivement doivent « voter »
    np.fill_diagonal(sim_pos, 0)
    denom = mask.dot(sim_pos)
    denom[denom == 0] = 1
    pred = mean_user[:, np.newaxis] + adj.dot(sim_pos) / denom
    return np.clip(pred, 1, 10)


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    """RMSE en ne tenant compte que des notes réellement présentes (valeurs non nulles)."""
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(sqrt(mean_squared_error(pred[mask], actual[mask])))


def cf_experiment(
    ratings: pd.DataFrame,
    n_users: int = 1500,
    n_animes: int = 1500,
    test_size: float = 0.2,
    seed: int = SEED,
) -> pd.DataFrame:
    """Expérience CF complète : split train/test, similarités user/item, RMSE de chaque approche.

    Contrairement à la version brute du notebook (corrélation sur les 3 colonnes brutes,
    puis sur des vecteurs remplis de 0), on construit la vraie matrice users x animes et
    on utilise un cosinus **ajusté** (centré par la moyenne de chaque utilisateur).
    """
    rated = rated_only(ratings)

    top_users = rated["user_id"].value_counts().head(n_users).index
    top_animes = rated["anime_id"].value_counts().head(n_animes).index
    subset = rated[rated["user_id"].isin(top_users) & rated["anime_id"].isin(top_animes)]

    train, test = train_test_split(subset, test_size=test_size, random_state=seed)

    users = np.array(sorted(subset["user_id"].unique()))
    animes = np.array(sorted(subset["anime_id"].unique()))
    u_pos = {u: i for i, u in enumerate(users)}
    a_pos = {a: i for i, a in enumerate(animes)}

    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        M = np.zeros((len(users), len(animes)))
        for u, a, r in df[["user_id", "anime_id", "rating"]].itertuples(index=False):
            M[u_pos[u], a_pos[a]] = r
        return M

    train_M, test_M = to_matrix(train), to_matrix(test)
    mask = train_M > 0
    mean_user = np.divide(
        train_M.sum(axis=1), mask.sum(axis=1),
        out=np.zeros(len(users)), where=mask.sum(axis=1) > 0,
    )
    adj = np.where(mask, train_M - mean_user[:, np.newaxis], 0.0)

    user_sim = _cosine(adj)          # users comparés sur leurs écarts à leur moyenne
    item_sim = _cosine(adj.T)        # cosinus ajusté « classique » pour les items
    np.fill_diagonal(user_sim, 0)
    np.fill_diagonal(item_sim, 0)

    rmse_user = rmse(predict_user_based(train_M, user_sim), test_M)
    rmse_item = rmse(predict_item_based(train_M, item_sim), test_M)
    baseline = float(test_M[test_M > 0].mean())  # prédiction = moyenne globale

    def rmse_of(value: float) -> float:
        err = test_M[test_M > 0] - value
        return float(sqrt(np.mean(err**2)))

    return pd.DataFrame(
        {
            "approche": ["user-based CF (cosinus ajusté)", "item-based CF (cosinus ajusté)", "baseline (moyenne globale)"],
            "RMSE": [rmse_user, rmse_item, rmse_of(baseline)],
        }
    ).set_index("approche")


# ---------------------------------------------------------------------------
# 5. Évaluation des stratégies (precision@k, couverture)
# ---------------------------------------------------------------------------

def precision_at_k(recos: list[int], aimes: set[int], k: int = 10) -> float:
    """Part des k premières recommandations réellement aimées par l'utilisateur."""
    if not recos:
        return 0.0
    return len(set(recos[:k]) & aimes) / k


def build_eval_split(ratings: pd.DataFrame, sample_size: int = 50_000, seed: int = SEED):
    """Split train/test d'un échantillon de notes + ensemble des animés aimés cachés par utilisateur."""
    small = rated_only(ratings).sample(n=min(sample_size, len(ratings)), random_state=seed)
    train, test = train_test_split(small, test_size=0.2, random_state=seed)
    aimes_test = test[test["rating"] >= LIKE_THRESHOLD].groupby("user_id")["anime_id"].apply(set)
    return train, aimes_test


def evaluate_strategies(
    strategies: dict[str, callable],
    train: pd.DataFrame,
    aimes_test: pd.Series,
    k: int = 10,
) -> pd.DataFrame:
    """precision@k, taux de réussite et couverture du catalogue pour chaque stratégie.

    Chaque stratégie a la signature (user_id, notes_train) -> liste d'anime_id.
    precision@k : moyenne par utilisateur (plafonnée par le faible nombre d'animés aimés cachés) ;
    réussite@k  : part d'utilisateurs ayant au moins 1 animé aimé dans le top k.
    """
    bilan = []
    for nom, recommander in strategies.items():
        recos = {u: recommander(u, train) for u in aimes_test.index}
        precisions = [precision_at_k(recos[u], aimes_test[u], k) for u in aimes_test.index]
        bilan.append(
            {
                "stratégie": nom,
                f"precision@{k}": round(float(np.mean(precisions)), 3),
                f"réussite@{k} (%)": round(100 * float(np.mean([p > 0 for p in precisions])), 1),
                "animes différents recommandés": len({m for liste in recos.values() for m in liste}),
            }
        )
    return pd.DataFrame(bilan).set_index("stratégie")
