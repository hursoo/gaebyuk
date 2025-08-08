# modules/scraping.py
from __future__ import annotations
import time
import pandas as pd
from typing import Iterable, Optional
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; KebyeokScraper/1.0; +https://example.org)"
}

def _make_session(timeout: int = 20, total_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """requests.Session with retry/backoff(재시도/백오프)."""
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request = _wrap_timeout(s.request, timeout=timeout)  # type: ignore
    return s

def _wrap_timeout(request_fn, timeout: int = 20):
    def _req(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return request_fn(method, url, **kwargs)
    return _req

def extract_content_from_html(html: str, parser: str = "html.parser") -> str:
    """페이지에서 div#cont_view 안의 텍스트만 추출."""
    soup = BeautifulSoup(html, parser)
    blocks = soup.find_all("div", {"id": "cont_view"})
    texts = []
    for b in blocks:
        texts.append(b.get_text("\n", strip=True))
    return "\n\n".join(texts).strip()

def scrape_contents(
    urls: Iterable[str],
    limit: Optional[int] = None,
    sleep_sec: float = 0.2,
    parser: str = "html.parser",
) -> pd.DataFrame:
    """
    urls: 기사 원문 페이지 URL 목록
    limit: 앞에서부터 n개만 수집(테스트용)
    return: DataFrame[r_id, content]
    """
    sess = _make_session()
    out = []
    for i, url in enumerate(urls):
        if limit is not None and i >= limit:
            break
        try:
            r = sess.get(url, verify=True)  # 인증서 검증(가능하면 True 유지)
            r.raise_for_status()
            content = extract_content_from_html(r.text, parser=parser)
            # 원본 노트북과 동일하게 'r_id'를 URL 끝 16자로 구성
            r_id = url[-16:]
            out.append((r_id, content))
        except Exception as e:
            out.append((None, f"[ERROR] {url} :: {e}"))
        time.sleep(sleep_sec)  # rate limiting(요청 간격)
    return pd.DataFrame(out, columns=["r_id", "content"])

def join_with_meta(meta_df: pd.DataFrame, contents_df: pd.DataFrame,
                   left_on: str = "r_id_raw", right_on: str = "r_id",
                   keep_cols: list[str] = None) -> pd.DataFrame:
    """
    메타데이터 DF와 본문 DF를 병합.
    keep_cols: 최종 보존 컬럼 순서 지정.
    """
    if keep_cols is None:
        keep_cols = ["r_id", "r_id_raw", "title", "writer", "gisa_class", "date", "url", "year", "content"]
    # 병합 전 중복 키 제거/정리는 필요에 따라 수행
    meta_df = meta_df.copy()
    if "r_id" in meta_df.columns and left_on != "r_id":
        # 원본 노트북처럼 r_id 중복 회피용으로 한쪽 열 드랍 필요하면 여기서 처리
        pass
    combi = pd.merge(meta_df, contents_df, left_on=left_on, right_on=right_on, how="inner")
    # 안전하게 컬럼 존재 확인 후 선택
    cols = [c for c in keep_cols if c in combi.columns]
    return combi[cols]
