# 📊 (핵심) 'AI 팩토리 중앙 관제소' (수동 제어 + 모니터링)
# ----------------------------------------------------
# 1. 3개의 AI 봇을 수동으로 즉시 실행 (데모용)
# 2. 봇이 생성한 데이터(CSV)를 확인
# 3. 봇이 남긴 로그(LOG)를 실시간으로 확인
# ----------------------------------------------------

import streamlit as st
import pandas as pd
import os
import subprocess # 봇을 백그라운드에서 실행
import time

# --- 1. 경로 설정 (NCP 서버의 절대 경로) ---
BASE_PATH = "/root/PII-Guardian" 
CRAWLER_SCRIPT = os.path.join(BASE_PATH, "crawler.py")
LABELER_SCRIPT = os.path.join(BASE_PATH, "autolabeler.py")
TRAIN_SCRIPT = os.path.join(BASE_PATH, "train.py")

DETECTED_FILE = os.path.join(BASE_PATH, "detected_leaks.csv")
FEEDBACK_FILE = os.path.join(BASE_PATH, "feedback_data.csv")

LOG_FILES = {
    "Crawler Log (신입 봇)": os.path.join(BASE_PATH, "crawler.log"),
    "Labeler Log (전문가 봇)": os.path.join(BASE_PATH, "autolabeler.log"),
    "Train Log (학습기)": os.path.join(BASE_PATH, "train.log")
}

# --- 2. 봇 실행 함수 (✨ Blocker 2 해결) ---
def run_script(script_path):
    """스크립트를 '논블로킹(non-blocking)' 방식으로 백그라운드에서 실행합니다."""
    
    # (✨ 핵심 수정)
    # 1. /usr/bin/python3 (시스템) -> {BASE_PATH}/venv/bin/python3 (가상환경)로 변경
    # 2. 로그가 '실시간 로그' 탭에 보이도록 Crontab과 동일하게 로그 파일로 리디렉션
    
    python_executable = os.path.join(BASE_PATH, "venv/bin/python3")
    log_file = script_path.replace('.py', '.log') # 예: crawler.py -> crawler.log
    
    # (중요) venv 파이썬이 존재하는지 확인
    if not os.path.exists(python_executable):
        st.error(f"❌ 실행 실패: 가상 환경({python_executable})을 찾을 수 없습니다.")
        st.error("deploy.yml이 venv를 생성했는지 확인하세요.")
        return

    try:
        # (nohup과 &를 사용해 대시보드가 꺼져도 봇이 계속 돌게 함)
        # (로그 파일에 표준 출력(>>)과 표준 에러(2>&1)를 모두 저장)
        command = f"nohup {python_executable} {script_path} >> {log_file} 2>&1 &"
        
        subprocess.Popen(command, shell=True)
        st.success(f"✅ {script_path.split('/')[-1]} 백그라운드 실행 시작!")
        st.info(f"결과는 10초 뒤 '실시간 로그' 탭 ({log_file.split('/')[-1]})에서 확인하세요.")
    except Exception as e:
        st.error(f"❌ 실행 실패: {e}")

# --- 3. 로그 읽기 함수 ---
def read_log_file(log_path):
    """로그 파일의 최신 100줄을 읽어옵니다."""
    if not os.path.exists(log_path):
        return f"로그 파일 없음: {log_path}\n(봇이 아직 한 번도 실행되지 않았거나, 경로 오류일 수 있습니다.)"
    try:
        with open(log_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return f"로그 파일이 비어있습니다: {log_path}"
            return "".join(lines[-100:]) # 최신 100줄만
    except Exception as e:
        return f"로그 읽기 오류: {e}"

# --- 4. 데이터 읽기 함수 (캐싱 사용) ---
@st.cache_data(ttl=10) # (수정) 60초 -> 10초로 줄여 더 실시간처럼 보이게 함
def load_csv(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame() # 빈 파일일 경우
    return pd.DataFrame()

# --- 5. Streamlit UI (웹페이지) ---
st.set_page_config(layout="wide")
st.title("🤖 AI 팩토리 중앙 관제소")
st.write(f"'{BASE_PATH}'에서 실행 중...")

# --- 3개의 탭으로 기능 분리 ---
tab1, tab2, tab3 = st.tabs(["🕹️ 수동 제어 (On-Demand)", "📊 데이터 뷰어", "📜 실시간 로그"])

# --- 탭 1: 수동 제어 버튼 ---
with tab1:
    st.header("🕹️ AI 팩토리 수동 실행")
    st.warning("Crontab이 자동으로 실행하지만, 지금 당장 테스트/데모가 필요할 때 사용하세요.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. '신입' 봇 (크롤러)")
        st.write("'의심' 목록 수집 (1분 소요)")
        if st.button("Start Crawler Now"):
            run_script(CRAWLER_SCRIPT)
            time.sleep(1) # 버튼 클릭 후 새로고침 시간 확보
            st.rerun()
            
    with col2:
        st.subheader("2. '전문가' 봇 (라벨러)")
        st.write("'의심' 목록 -> '정답' 생성 (N분 소요)")
        if st.button("Start Auto-Labeler Now"):
            run_script(LABELER_SCRIPT)
            time.sleep(1)
            st.rerun()

    with col3:
        st.subheader("3. '학습기' (트레이너)")
        st.write("'정답' -> '경력직 뇌' 훈련 (30초 시뮬레이션)")
        if st.button("Start Training Now"):
            run_script(TRAIN_SCRIPT)
            time.sleep(1)
            st.rerun()

# --- 탭 2: 데이터 뷰어 (읽기 전용) ---
with tab2:
    st.header("📊 데이터 뷰어")
    if st.button("데이터 새로고침"):
        st.cache_data.clear() # 캐시 비우기
        st.rerun()
        
    st.subheader(f"📝 '신입' 봇이 수집한 '의심' 목록 ({DETECTED_FILE})")
    df_detected = load_csv(DETECTED_FILE)
    st.dataframe(df_detected, use_container_width=True)
        
    st.subheader(f"✅ '전문가' 봇이 만든 '정답' 목록 ({FEEDBACK_FILE})")
    df_feedback = load_csv(FEEDBACK_FILE)
    st.dataframe(df_feedback, use_container_width=True)

# --- 탭 3: 실시간 로그 뷰어 ---
with tab3:
    st.header("📜 실시간 로그 뷰어")
    if st.button("로그 새로고침"):
        st.rerun()
    
    for log_name, log_path in LOG_FILES.items():
        st.subheader(log_name)
        log_content = read_log_file(log_path)
        st.text_area(f"Log: {log_path}", log_content, height=300, key=log_path)

