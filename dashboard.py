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
README_FILE = os.path.join(BASE_PATH, "README.md")

LOG_FILES = {
    "Crawler Log (신입 봇)": os.path.join(BASE_PATH, "crawler.log"),
    "Labeler Log (전문가 봇)": os.path.join(BASE_PATH, "autolabeler.log"),
    "Train Log (학습기)": os.path.join(BASE_PATH, "train.log")
}

# --- 2. 봇 실행 함수 (✨ Blocker 2 해결) ---
def run_script(script_path):
    """스크립트를 '논블로킹(non-blocking)' 방식으로 백그라운드에서 실행합니다."""
    
    python_executable = os.path.join(BASE_PATH, "venv/bin/python3")
    log_file = script_path.replace('.py', '.log') # 예: crawler.py -> crawler.log
    
    if not os.path.exists(python_executable):
        st.error(f"❌ 실행 실패: 가상 환경({python_executable})을 찾을 수 없습니다.")
        st.error("deploy.yml이 venv를 생성했는지 확인하세요.")
        return

    try:
        command = f"nohup {python_executable} {script_path} >> {log_file} 2>&1 &"
        
        subprocess.Popen(command, shell=True)
        st.success(f"✅ {script_path.split('/')[-1]} 백그라운드 실행 시작!")
        st.info(f"결과는 10초 뒤 '실시간 로그' 탭 ({log_file.split('/')[-1]})에서 확인하세요.")
    except Exception as e:
        st.error(f"❌ 실행 실패: {e}")

# --- 3. 로그 읽기 함수 (✨ 캐시 문제 해결) ---
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

# --- 4. 데이터 읽기 함수 (캐싱 사용) ---
@st.cache_data(ttl=10) # 10초마다 데이터 새로고침
def load_csv(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame() # 빈 파일일 경우
    return pd.DataFrame()

# --- (✨ 신규) README 마크다운 로드 ---
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

# --- (✨ 수정) 4개의 탭으로 기능 분리 ---
tab_overview, tab_control, tab_hitl, tab_logs = st.tabs([
    "🏠 개요", 
    "🕹️ 수동 제어 (On-Demand)", 
    "📊 데이터 뷰어 및 수정 (HITL)", 
    "📜 실시간 로그"
])

# --- 탭 0: 개요 (README) ---
with tab_overview:
    st.header("프로젝트 개요")
    st.markdown(load_readme(), unsafe_allow_html=True)

# --- 탭 1: 수동 제어 버튼 ---
with tab_control:
    st.header("🕹️ AI 팩토리 수동 실행")
    st.warning("Crontab이 자동으로 실행하지만, 지금 당장 테스트/데모가 필요할 때 사용하세요.")
    
    # (✨ 신규) 실시간 현황판
    st.subheader("📈 실시간 현황")
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

    # 봇 실행 버튼
    st.header("⚙️ 봇 실행기")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. '신입' 봇 (크롤러)")
        st.write("'의심' 목록 수집 (1분 소요)")
        if st.button("Start Crawler Now 🕵️"):
            run_script(CRAWLER_SCRIPT)
            time.sleep(1) 
            st.cache_data.clear()
            st.rerun()
            
    with col2:
        st.subheader("2. '전문가' 봇 (라벨러)")
        st.write("'의심' 목록 -> '정답' 생성 (N분 소요)")
        if st.button("Start Auto-Labeler Now 🧑‍🏫"):
            run_script(LABELER_SCRIPT)
            time.sleep(1)
            st.cache_data.clear()
            st.rerun()

    with col3:
        st.subheader("3. '학습기' (트레이너)")
        st.write("'정답' -> '경력직 뇌' 훈련 (30초 시뮬)")
        if st.button("Start Training Now 🎓"):
            run_script(TRAIN_SCRIPT)
            time.sleep(1)
            st.cache_data.clear()
            st.rerun()

# --- 탭 2: (✨ 수정) 데이터 뷰어 및 수정 (HITL) ---
with tab_hitl:
    st.header("📊 데이터 뷰어 및 수정 (Human-in-the-Loop)")
    st.info("AI가 잘못 판단한 경우, 'llm_label'을 직접 수정하고 '변경사항 저장' 버튼을 누르세요.")

    if st.button("데이터 새로고침 🔄"):
        st.cache_data.clear()
        st.rerun()
        
    st.subheader(f"✅ '누적 정답' 목록 ({FEEDBACK_FILE})")
    
    if 'feedback_df' not in st.session_state:
        st.session_state.feedback_df = load_csv(FEEDBACK_FILE)

    # (✨ 핵심) 수정 가능한 데이터 에디터 사용
    edited_df = st.data_editor(
        st.session_state.feedback_df,
        num_rows="dynamic",
        use_container_width=True,
        # '유출', '공개', '오류' 외에는 선택 못하게 막기
        column_config={
            "llm_label": st.column_config.SelectboxColumn(
                "LLM Label",
                help="AI의 판단 (유출/공개). 여기서 수정 가능!",
                options=["유출", "공개", "오류"],
                required=True,
            )
        }
    )

    if st.button("변경사항 저장 💾", type="primary"):
        try:
            edited_df.to_csv(FEEDBACK_FILE, index=False, encoding='utf-8-sig')
            st.session_state.feedback_df = edited_df
            st.success("✅ 변경사항이 feedback_data.csv에 성공적으로 저장되었습니다!")
            # 다른 탭의 캐시도 비워줌
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"❌ 저장 실패: {e}")

    st.divider()
    
    st.subheader(f"📝 '처리 대기' 목록 ({DETECTED_FILE}) - (읽기 전용)")
    df_detected_readonly = load_csv(DETECTED_FILE)
    st.dataframe(df_detected_readonly, use_container_width=True)


# --- 탭 3: (✨ 수정) 실시간 로그 뷰어 (Selectbox) ---
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