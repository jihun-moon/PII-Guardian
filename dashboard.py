# 📊 (핵심) 'AI 팩토리 중앙 관제소' (모니터링 전용)
# (v2.0 - 수동 제어 버튼 삭제, 스케줄 뷰어 추가, 저장 버튼 삭제)
# ----------------------------------------------------
# 1. 봇이 생성한 데이터(CSV)를 확인
# 2. 봇이 남긴 로그(LOG)를 실시간으로 확인
# 3. 봇의 자동 실행 스케줄(Crontab)을 확인
# ----------------------------------------------------

import streamlit as st
import pandas as pd
import os

# --- 1. 경로 설정 (NCP 서버의 절대 경로) ---
BASE_PATH = "/root/PII-Guardian" 
# (수동 실행 스크립트 경로 제거)

DETECTED_FILE = os.path.join(BASE_PATH, "detected_leaks.csv")
FEEDBACK_FILE = os.path.join(BASE_PATH, "feedback_data.csv")
README_FILE = os.path.join(BASE_PATH, "README.md")

LOG_FILES = {
    "Crawler Log (신입 봇)": os.path.join(BASE_PATH, "crawler.log"),
    "Labeler Log (전문가 봇)": os.path.join(BASE_PATH, "autolabeler.log"),
    "Train Log (학습기)": os.path.join(BASE_PATH, "train.log")
}

# --- 2. 봇 실행 함수 (✨ v2.0: 삭제) ---
# (run_script 함수 전체 삭제)

# --- 3. 로그 읽기 함수 (기존과 동일) ---
@st.cache_data(ttl=5)
def read_log_file(log_path):
    """로그 파일의 최신 100줄을 읽어옵니다."""
    if not os.path.exists(log_path):
        return f"로그 파일 없음: {log_path}\n(봇이 아직 한 번도 실행되지 않았거나, 경로 오류일 수 있습니다.)"
    try:
        with open(log_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return f"로그 파일이 비어있습니다: {log_path}\n(봇이 실행 중이거나, 방금 실행을 시작했을 수 있습니다.)"
            return "".join(lines[-100:]) # 최신 100줄만
    except Exception as e:
        return f"로그 읽기 오류: {e}"

# --- 4. 데이터 읽기 함수 (기존과 동일) ---
@st.cache_data(ttl=10) # 10초마다 데이터 새로고침
def load_csv(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame() # 빈 파일일 경우
    return pd.DataFrame()

# --- (신규) README 마크다운 로드 (기존과 동일) ---
@st.cache_data
def load_readme():
    if os.path.exists(README_FILE):
        with open(README_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return "README.md 파일을 찾을 수 없습니다."

# --- 5. Streamlit UI (웹페이지) ---
st.set_page_config(page_title="PII-Guardian", layout="wide", page_icon="🤖")
st.title("🤖 PII-Guardian: AI 팩토리 관제소")
st.write(f"'{BASE_PATH}'에서 실행 중...")

# --- (✨ v2.0: 탭 이름 변경) ---
tab_overview, tab_status, tab_data_viewer, tab_logs = st.tabs([
    "🏠 개요", 
    "📈 팩토리 현황 및 스케줄", 
    "📊 데이터 뷰어 (읽기 전용)", 
    "📜 실시간 로그"
])

# --- 탭 0: 개요 (README) ---
with tab_overview:
    st.header("프로젝트 개요")
    st.markdown(load_readme(), unsafe_allow_html=True)

# --- 탭 1: (✨ v2.0: '수동 제어' -> '현황 및 스케줄') ---
with tab_status:
    st.header("📈 AI 팩토리 실시간 현황")
    
    # 실시간 현황판 (기존과 동일)
    col_metric1, col_metric2 = st.columns(2)
    df_detected = load_csv(DETECTED_FILE)
    df_feedback = load_csv(FEEDBACK_FILE)
    
    col_metric1.metric(
        label="🕵️ 처리 대기 ('신입' 봇 발견)", 
        value=f"{len(df_detected)} 건",
        help="crawler.py가 발견하여 detected_leaks.csv에 쌓인 '의심' 목록입니다."
    )
    col_metric2.metric(
        label="✅ 누적 처리 완료 ('전문가' 봇 판단)", 
        value=f"{len(df_feedback)} 건",
        help="autolabeler.py가 HyperCLOVA에 물어보고 feedback_data.csv에 누적한 '정답' 목록입니다."
    )
    
    if st.button("현황판 새로고침 🔄"):
        st.cache_data.clear()
        st.rerun()
        
    st.divider()

    # (✨✨✨ v2.0: '봇 실행기' 삭제 -> '스케줄 뷰어'로 변경) ---
    st.header("⚙️ 봇 자동 실행 스케줄 (Crontab)")
    st.info("이 봇들은 서버에 설정된 Crontab 스케줄에 따라 자동으로 실행됩니다.")
    
    # (참고: 이 스케줄은 이전에 crontab에 설정한 내용 기준입니다)
    schedule_data = {
        "봇 이름": [
            "1. '신입' 봇 (Crawler)", 
            "2. '전문가' 봇 (AutoLabeler)", 
            "3. '학습기' (Trainer)"
        ],
        "스케줄 (Cron)": [
            "0 * * * *", 
            "0 1 * * *", 
            "0 2 * * *"
        ],
        "실행 주기": [
            "매시 0분 (1시간마다)", 
            "매일 새벽 1시 0분", 
            "매일 새벽 2시 0분"
        ],
        "담당 파일": [
            "crawler.py", 
            "autolabeler.py", 
            "train.py"
        ]
    }
    st.dataframe(pd.DataFrame(schedule_data).set_index("봇 이름"), use_container_width=True)


# --- 탭 2: (✨ v2.0: '데이터 뷰어 및 수정' -> '데이터 뷰어 (읽기 전용)') ---
with tab_data_viewer:
    st.header("📊 데이터 뷰어 (읽기 전용)")
    
    # (✨ v2.0: info 텍스트 변경)
    st.info("AI 봇이 판독한 '누적 정답' 목록입니다. (읽기 전용)")

    if st.button("데이터 새로고침 🔄"):
        st.cache_data.clear()
        st.rerun()
        
    st.subheader(f"✅ '누적 정답' 목록 ({FEEDBACK_FILE})")
    
    # (✨✨✨ v2.0: 'st.data_editor' -> 'st.dataframe'으로 변경하여 읽기 전용으로)
    df_feedback_readonly = load_csv(FEEDBACK_FILE)
    st.dataframe(df_feedback_readonly, use_container_width=True)

    # (✨✨✨ v2.0: '변경사항 저장' 버튼 삭제) ---
    # (st.button("변경사항 저장 💾", ...) 블록 전체 삭제)

    st.divider()
    
    st.subheader(f"📝 '처리 대기' 목록 ({DETECTED_FILE}) - (읽기 전용)")
    df_detected_readonly = load_csv(DETECTED_FILE)
    st.dataframe(df_detected_readonly, use_container_width=True)


# --- 탭 3: 실시간 로그 뷰어 (기존과 동일) ---
with tab_logs:
    st.header("📜 실시간 로그 뷰어")
    st.write("✨ (참고) 이 탭은 5초마다 자동으로 새로고침됩니다.")
    
    # 로그 파일 선택
    log_choice_name = st.selectbox("표시할 로그 파일을 선택하세요:", LOG_FILES.keys())
    
    if st.button("로그 즉시 새로고침 🔄"):
        st.cache_data.clear()
        st.rerun()
    
    # 선택된 로그 표시
    if log_choice_name:
        log_path = LOG_FILES[log_choice_name]
        log_content = read_log_file(log_path)
        st.text_area(f"Log: {log_path}", log_content, height=400, key=log_path)