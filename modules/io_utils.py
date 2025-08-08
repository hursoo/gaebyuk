import pandas as pd
import streamlit as st
import re
from google.oauth2.service_account import Credentials
import gspread

@st.cache_resource(show_spinner=False)
def get_gs_client():
    info = st.secrets.get("gcp_service_account", None)
    if info is None:
        raise RuntimeError("secrets에 gcp_service_account가 없습니다. 서비스 계정 JSON을 저장하세요.")
    creds = Credentials.from_service_account_info(info, scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
    client = gspread.authorize(creds)
    return client

def read_google_sheet(sheet_url: str, worksheet_name: str | None = None) -> pd.DataFrame:
    client = get_gs_client()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        raise ValueError("유효한 Google Sheets URL이 아닙니다.")
    sheet_id = m.group(1)
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
    data = ws.get_all_records()
    return pd.DataFrame(data)
