# 개벽 텍스트 분석 워크플로우 (Streamlit)

단계형 파이프라인: 입력 → 정제 → 토큰화 → DTM/TF-IDF → 토픽모델링 → 해석 → 내보내기

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Sheets
Streamlit Cloud > App > Settings > Secrets에 서비스 계정 JSON을 `gcp_service_account` 키로 저장하고,
해당 서비스 계정 이메일을 Google Sheets에 공유하세요.
