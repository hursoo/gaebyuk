import streamlit as st
import pandas as pd
from modules import io_utils, pipeline
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import platform, os
from modules import scraping  # ← 추가


st.set_page_config(page_title="개벽 텍스트 분석 워크플로우", page_icon="📚", layout="wide")

st.title("📚 개벽 텍스트 분석 워크플로우 (모듈형)")
st.caption("로컬 파일 또는 Google Sheets를 입력으로 받아 정제 → 토큰화 → DTM/TF-IDF → 토픽모델링 → 해석까지 단계적으로 진행합니다.")

# ------------------------------------------------------------------------------------
# Sidebar (Step Navigator)
# ------------------------------------------------------------------------------------
steps = [
    "1) 데이터 입력",
    "2) 정제/전처리",
    "3) 토큰화(kiwi/폴백)",
    "4) DTM/TF-IDF",
    "5) 토픽모델링(LDA/DMR)",
    "6) 결과 탐색/시각화",
    "7) 결과 저장/내보내기"
]
current = st.sidebar.radio("단계 선택", steps, index=0)

# Global params on sidebar
st.sidebar.subheader("📐 전역 파라미터")
ngram_min = st.sidebar.number_input("n-gram 최소", 1, 5, 1)
ngram_max = st.sidebar.number_input("n-gram 최대", 1, 5, 2)
min_df = st.sidebar.number_input("min_df (문서 최소 등장수)", 1, 100, 2)
max_features = st.sidebar.number_input("최대 피처 수", 100, 100000, 5000, step=500)
use_idf = st.sidebar.checkbox("TF-IDF 사용", value=True)
remove_one_char = st.sidebar.checkbox("한 글자 토큰 제거", value=True)

# Session state containers
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "tokens_df" not in st.session_state:
    st.session_state.tokens_df = None
if "dtm" not in st.session_state:
    st.session_state.dtm = None
if "vocab" not in st.session_state:
    st.session_state.vocab = None
if "topic_model" not in st.session_state:
    st.session_state.topic_model = None
if "topic_result" not in st.session_state:
    st.session_state.topic_result = None

# ------------------------------------------------------------------------------------
# Utility: find Korean font
# ------------------------------------------------------------------------------------
def resolve_korean_font():
    local_font = Path(__file__).parent / "assets" / "NanumGothic.ttf"
    if local_font.exists():
        return str(local_font)
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates += ["C:/Windows/Fonts/malgun.ttf"]
    elif system == "Darwin":
        candidates += ["/System/Library/Fonts/Supplemental/AppleGothic.ttf"]
    else:
        candidates += [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

font_path = resolve_korean_font()

# ------------------------------------------------------------------------------------
# Step 1: 데이터 입력
# ------------------------------------------------------------------------------------
if current == steps[0]:
    st.header("1) 데이터 입력")
    st.write("엑셀(.xlsx) 업로드 또는 Google Sheets에서 불러오기(서비스 계정 필요)")

    tab1, tab2, tab3 = st.tabs(["엑셀 업로드", "Google Sheets", "웹 스크래핑"])

    with tab1:
        up = st.file_uploader("엑셀(.xlsx) 업로드", type=["xlsx"])
        sheet_name = st.text_input("시트 이름 (비워두면 첫 시트)", "")
        if st.button("엑셀 불러오기"):
            if up is None:
                st.warning("파일을 업로드하세요.")
            else:
                try:
                    df = pd.read_excel(up, sheet_name=sheet_name if sheet_name else 0, engine="openpyxl")
                    st.session_state.raw_df = df
                    st.success(f"불러오기 완료: {df.shape}")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"엑셀 읽기 실패: {e}")

    with tab2:
        gs_url = st.text_input("Google Sheets URL")
        ws_name = st.text_input("워크시트(탭) 이름", "")
        st.caption("⚠️ Streamlit Cloud에서 Google Sheets를 사용하려면 'secrets'에 서비스 계정 JSON을 저장하고, 해당 이메일을 시트 공유에 추가해야 합니다.")
        if st.button("시트 불러오기"):
            try:
                df = io_utils.read_google_sheet(gs_url, ws_name or None)
                st.session_state.raw_df = df
                st.success(f"불러오기 완료: {df.shape}")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"시트 읽기 실패: {e}")

    with tab3:
        st.markdown("**엑셀/시트의 URL 열을 사용하지 않고, 바로 URL 목록을 입력해 스크래핑합니다.**")
        urls_text = st.text_area("URL 목록(줄바꿈으로 구분)", height=150,
                                 placeholder="https://db.history.go.kr/.../rId=XXXXXXXXXXXXXXX\nhttps://db.history.go.kr/.../rId=YYYYYYYYYYYYYYY")
        limit = st.number_input("앞에서부터 n개만 수집(테스트용)", 1, 10000, 10)
        if st.button("스크래핑 실행"):
            urls_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not urls_list:
                st.warning("URL을 한 줄에 하나씩 입력하세요.")
            else:
                contents_df = scraping.scrape_contents(urls_list, limit=limit)
                st.success(f"스크래핑 완료: {contents_df.shape}")
                st.dataframe(contents_df.head(10))
                # 메타와 병합이 필요하면, 탭1/탭2에서 읽어온 메타 DF(st.session_state.raw_df)를 활용
                if st.session_state.raw_df is not None:
                    merged = scraping.join_with_meta(
                        st.session_state.raw_df.copy(),
                        contents_df,
                        left_on="r_id_raw", right_on="r_id",
                        keep_cols=["r_id", "r_id_raw", "title", "writer", "gisa_class", "date", "url", "year", "content"]
                    )
                    st.subheader("메타+본문 병합 결과")
                    st.dataframe(merged.head(10))
                    # 다음 단계에서 쓰도록 저장
                    st.session_state.raw_df = merged
                else:
                    # 다음 단계에서 정제/토큰화로 쓰게 매핑
                    st.session_state.raw_df = contents_df


# ------------------------------------------------------------------------------------
# Step 2: 정제/전처리
# ------------------------------------------------------------------------------------
if current == steps[1]:
    st.header("2) 정제/전처리")
    if st.session_state.raw_df is None:
        st.info("먼저 '1) 데이터 입력'에서 데이터를 불러오세요.")
    else:
        text_col = st.selectbox("텍스트가 들어있는 열 선택", st.session_state.raw_df.columns.tolist())
        normalize_space = st.checkbox("공백 정규화", value=True)
        lower = st.checkbox("소문자화(영문)", value=False)
        remove_punct = st.checkbox("기본 구두점 제거", value=False)

        if st.button("정제 실행"):
            clean_df = pipeline.clean_texts(
                st.session_state.raw_df.copy(),
                text_col=text_col,
                normalize_space=normalize_space,
                to_lower=lower,
                remove_punct=remove_punct,
            )
            st.session_state.clean_df = clean_df
            st.success(f"정제 완료: {clean_df.shape}")
            st.dataframe(clean_df.head(10))

# ------------------------------------------------------------------------------------
# Step 3: 토큰화
# ------------------------------------------------------------------------------------
if current == steps[2]:
    st.header("3) 토큰화(kiwi 우선, soynlp 폴백)")
    if st.session_state.clean_df is None:
        st.info("먼저 '2) 정제/전처리'를 완료하세요.")
    else:
        text_col = st.selectbox("토큰화할 열 선택", st.session_state.clean_df.columns.tolist())
        use_pos = st.checkbox("품사 태깅 사용", value=True)
        pos_keep = st.multiselect("유지할 품사", ['NNG', 'NNP', 'VV', 'VA'], default=['NNG', 'NNP'])
        if st.button("토큰화 실행"):
            tokens_df, used = pipeline.tokenize_texts(
                st.session_state.clean_df.copy(),
                text_col=text_col,
                pos_keep=pos_keep if use_pos else None,
                remove_one_char=remove_one_char,
            )
            st.session_state.tokens_df = tokens_df
            st.success(f"토큰화 완료 (엔진: {used})")
            st.dataframe(tokens_df.head(10))

        if st.session_state.tokens_df is not None:
            st.subheader("상위 빈도 토큰")
            all_tokens = [t for row in st.session_state.tokens_df["tokens"] for t in row]
            top = Counter(all_tokens).most_common(30)
            st.write(top)
            if font_path:
                try:
                    wc = WordCloud(font_path=font_path, background_color="white", width=900, height=500)
                    cloud = wc.generate_from_frequencies(dict(top))
                    fig, ax = plt.subplots()
                    ax.imshow(cloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"워드클라우드 실패: {e}")

# ------------------------------------------------------------------------------------
# Step 4: DTM/TF-IDF
# ------------------------------------------------------------------------------------
if current == steps[3]:
    st.header("4) DTM/TF-IDF 생성")
    if st.session_state.tokens_df is None:
        st.info("먼저 '3) 토큰화'를 완료하세요.")
    else:
        if st.button("DTM/TF-IDF 만들기"):
            dtm, vocab, vectorizer = pipeline.build_vector(
                st.session_state.tokens_df,
                ngram_range=(int(ngram_min), int(ngram_max)),
                min_df=int(min_df),
                max_features=int(max_features),
                use_idf=use_idf
            )
            st.session_state.dtm = dtm
            st.session_state.vocab = vocab
            st.success(f"DTM/TF-IDF 완료: shape={dtm.shape}, vocab={len(vocab)}")
            st.write("상위 피처:", list(vocab)[:50])

# ------------------------------------------------------------------------------------
# Step 5: 토픽모델링
# ------------------------------------------------------------------------------------
if current == steps[4]:
    st.header("5) 토픽모델링 (LDA / DMR 예정)")
    if st.session_state.dtm is None:
        st.info("먼저 '4) DTM/TF-IDF'를 완료하세요.")
    else:
        num_topics = st.number_input("토픽 수(k)", 2, 50, 8)
        max_iter = st.number_input("반복 수", 10, 2000, 500, step=50)
        seed = st.number_input("시드", 0, 100000, 1000, step=1)
        algo = st.selectbox("알고리즘", ["sklearn-LDA", "tomotopy-LDA (실험적)"], index=0)

        if st.button("토픽 학습"):
            model, result = pipeline.run_topic_model(
                st.session_state.dtm,
                st.session_state.vocab,
                num_topics=int(num_topics),
                max_iter=int(max_iter),
                seed=int(seed),
                algo=algo
            )
            st.session_state.topic_model = model
            st.session_state.topic_result = result
            st.success("토픽 학습 완료")

# ------------------------------------------------------------------------------------
# Step 6: 결과 탐색
# ------------------------------------------------------------------------------------
if current == steps[5]:
    st.header("6) 결과 탐색/시각화")
    if st.session_state.topic_result is None:
        st.info("먼저 '5) 토픽모델링'을 완료하세요.")
    else:
        topn = st.slider("토픽별 상위 단어 수", 5, 30, 10)
        topics = pipeline.get_top_words(st.session_state.topic_result, topn=topn)
        st.subheader("토픽별 상위 단어")
        st.dataframe(pd.DataFrame({f"Topic {i}": words for i, words in enumerate(topics)}))

# ------------------------------------------------------------------------------------
# Step 7: 결과 저장
# ------------------------------------------------------------------------------------
if current == steps[6]:
    st.header("7) 결과 저장/내보내기")
    if st.session_state.topic_result is None:
        st.info("먼저 '5) 토픽모델링'을 완료하세요.")
    else:
        csv = pipeline.export_topics_csv(st.session_state.topic_result)
        st.download_button("토픽 상위어 CSV 다운로드", data=csv, file_name="topics.csv", mime="text/csv")
