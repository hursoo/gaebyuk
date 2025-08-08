import pandas as pd
import numpy as np
import re, string
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def _try_import_tomotopy():
    try:
        import tomotopy as tp
        return tp
    except Exception:
        return None

# 2) 정제/전처리
def clean_texts(df: pd.DataFrame, text_col: str, normalize_space=True, to_lower=False, remove_punct=False) -> pd.DataFrame:
    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = "" if pd.isna(s) else str(s)
        if to_lower:
            s = s.lower()
        if remove_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        if normalize_space:
            s = re.sub(r"\s+", " ", s).strip()
        return s
    df = df.copy()
    df[text_col] = df[text_col].map(_clean)
    return df

# 3) 토큰화 (kiwi → soynlp fallback)
@st.cache_resource(show_spinner=False)
def _load_tokenizers():
    engine = None
    kiwi = None
    fallback = None
    reason = ""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        kiwi.analyze("테스트")
        engine = "kiwi"
    except Exception as e:
        reason = str(e)
        try:
            from soynlp.tokenizer import LTokenizer
            scores = {"한국":1.0,"역사":1.0,"분석":1.0,"텍스트":1.0}
            fallback = LTokenizer(scores)
            engine = "soynlp"
        except Exception as e2:
            reason += f" | soynlp 실패: {e2}"
    return engine, kiwi, fallback, reason

def tokenize_texts(df: pd.DataFrame, text_col: str, pos_keep=None, remove_one_char=True):
    engine, kiwi, fallback, reason = _load_tokenizers()
    tokens_col = []

    if engine == "kiwi":
        for s in df[text_col].fillna("").astype(str).tolist():
            analyzed = kiwi.analyze(s)
            if pos_keep:
                toks = [t.form for sent in analyzed for t in sent[0] if t.tag in set(pos_keep)]
            else:
                toks = [t.form for sent in analyzed for t in sent[0]]
            if remove_one_char:
                toks = [t for t in toks if len(t) > 1]
            tokens_col.append(toks)
    elif engine == "soynlp":
        for s in df[text_col].fillna("").astype(str).tolist():
            toks = fallback.tokenize(s)
            if remove_one_char:
                toks = [t for t in toks if len(t) > 1]
            tokens_col.append(toks)
    else:
        raise RuntimeError("토크나이저 로드 실패: " + reason)

    out = df.copy()
    out["tokens"] = tokens_col
    return out, engine

# 4) Vectorize
def build_vector(tokens_df: pd.DataFrame, ngram_range=(1,2), min_df=2, max_features=5000, use_idf=True):
    texts = [" ".join(tokens) for tokens in tokens_df["tokens"].tolist()]
    if use_idf:
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    else:
        vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    return X, vocab, vec

# 5) Topic Modeling
def run_topic_model(X, vocab, num_topics=8, max_iter=500, seed=1000, algo="sklearn-LDA"):
    if algo.startswith("sklearn"):
        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iter, learning_method="batch", random_state=seed)
        W = lda.fit_transform(X)
        H = lda.components_
        return lda, {"W": W, "H": H, "vocab": vocab, "algo": "sklearn-LDA"}
    else:
        tp = _try_import_tomotopy()
        if tp is None:
            raise RuntimeError("tomotopy 불가(미설치 또는 환경 미지원). 'sklearn-LDA'를 사용하세요.")
        raise NotImplementedError("tomotopy 경로는 교수님 Colab 코드를 붙여 구현하세요.")

# 6) Explore
def get_top_words(result, topn=10):
    H = result["H"]
    vocab = result["vocab"]
    top_words = []
    for k in range(H.shape[0]):
        idx = np.argsort(-H[k])[:topn]
        words = [vocab[i] for i in idx]
        top_words.append(words)
    return top_words

# 7) Export
def export_topics_csv(result):
    import csv
    from io import StringIO
    sio = StringIO()
    H = result["H"]
    vocab = result["vocab"]
    writer = csv.writer(sio)
    writer.writerow(["topic", "rank", "term", "weight"])
    for k in range(H.shape[0]):
        idx = np.argsort(-H[k])
        for r, i in enumerate(idx[:50]):
            writer.writerow([k, r+1, vocab[i], float(H[k, i])])
    return sio.getvalue()
