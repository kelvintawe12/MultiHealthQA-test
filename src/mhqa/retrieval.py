# mhqa/retrieval.py
"""
TF-IDF nearest-neighbour retrieval baseline.
PerLanguageRetriever fits a separate TF-IDF index per subset so that
retrieval is always within-language (prevents cross-lingual mismatches).

Usage:
    retr = PerLanguageRetriever().fit(train_df)
    preds, scores, sources = retr.predict(holdout_df)
    rouge = compute_rouge(preds, holdout_df['output'].tolist())
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mhqa.constants import QUESTION_COL, ANSWER_COL, SUBSET_COL, SUBSET_ORDER


class PerLanguageRetriever:
    """
    Fit one TF-IDF index per subset (language-country configuration).
    At prediction time, retrieves the nearest training question and returns
    its stored answer as the prediction.
    """

    def __init__(self, max_features: int = 20_000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self._indexes: dict = {}   # subset → (vectorizer, tfidf_matrix, answer_list)

    def fit(self, train_df: pd.DataFrame) -> "PerLanguageRetriever":
        for subset in SUBSET_ORDER:
            sub = train_df[train_df[SUBSET_COL] == subset]
            if sub.empty:
                continue
            vect = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
            )
            matrix  = vect.fit_transform(sub[QUESTION_COL].tolist())
            answers = sub[ANSWER_COL].tolist()
            self._indexes[subset] = (vect, matrix, answers)
        print(f"PerLanguageRetriever fitted on {len(self._indexes)} subset(s).")
        return self

    def predict(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Returns:
            predictions : list of retrieved answer strings
            scores      : cosine similarity of each match
            sources     : which subset index was used per row
        """
        predictions, scores, sources = [], [], []
        for _, row in df.iterrows():
            subset = row[SUBSET_COL]
            if subset not in self._indexes:
                predictions.append("")
                scores.append(0.0)
                sources.append(subset)
                continue
            vect, matrix, answers = self._indexes[subset]
            q_vec = vect.transform([row[QUESTION_COL]])
            sims  = cosine_similarity(q_vec, matrix).flatten()
            best  = int(np.argmax(sims))
            predictions.append(answers[best])
            scores.append(float(sims[best]))
            sources.append(subset)

        return predictions, scores, sources
