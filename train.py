# 🎓 (봇 3) '학습기' 봇. '자동' 정답으로 '신입' 봇 뇌 훈련 -> my-ner-model
# ----------------------------------------------------
# (✨ 최종 로직: Log 기반)
# 1. '정답' 목록 (feedback_data.csv) [In_2]를 읽습니다.
# 2. 'trained.log' (학습 기록)을 읽습니다.
# 3. [In_2]에만 있고 [학습 기록]에는 없는 "새로운 정답"만 학습합니다.
# 4. 학습 완료 후, "새로운 정답"의 ID를 'trained.log'에 '추가'합니다.
# ----------------------------------------------------

import pandas as pd
import os
import time
import datetime

# (✨ 경로 수정) BASE_PATH 기준으로 경로 재설정
BASE_PATH = "/root/PII-Guardian" 
FEEDBACK_FILE = os.path.join(BASE_PATH, 'feedback_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'my-ner-model')
TRAINED_LOG_FILE = os.path.join(BASE_PATH, 'trained.log')
LAST_TRAINED_FILE = os.path.join(MODEL_PATH, 'last_trained.txt') # 학습 완료 시간

def load_trained_log():
    """이미 학습한 항목(중복 학습 방지용)을 불러옵니다."""
    if not os.path.exists(TRAINED_LOG_FILE):
        return set()
    try:
        with open(TRAINED_LOG_FILE, 'r', encoding='utf-8') as f:
            # (content, url)을 합친 고유 ID를 set으로 저장
            return set(line.strip() for line in f)
    except Exception as e:
        print(f"⚠️ 'trained.log' 로드 실패: {e}")
        return set()

def save_trained_log(unique_id):
    """학습 완료된 항목을 기록합니다."""
    with open(TRAINED_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(unique_id + '\n')

def main():
    print("🤖 3. '학습기' 봇(Trainer) 작동 시작...")
    
    # 1. '정답' 파일(In_2)이 있는지 확인
    if not os.path.exists(FEEDBACK_FILE):
        print(f"⚠️ '정답' 목록({FEEDBACK_FILE})이 없습니다. 학습을 건너뜁니다.")
        return

    # 2. '정답' 파일과 '학습 기록' 로드
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        feedback_df['url'] = feedback_df['url'].fillna('N/A') # 키 값 비교를 위해 N/A 처리
    except pd.errors.EmptyDataError:
        print("✅ '정답' 목록이 비어있습니다. 학습을 건너뜁니다.")
        return
    except Exception as e:
        print(f"❌ '정답' 파일 로드 중 에러: {e}")
        return
        
    trained_set = load_trained_log()

    # 3. "새로운" '유출' 데이터만 필터링
    new_data_to_train = []
    new_ids_to_log = []
    
    for index, row in feedback_df.iterrows():
        # (content, url)로 고유 ID 생성
        unique_id = f"{row['content']}|{row['url']}"
        
        # (조건 1) "유출" 라벨이고, (조건 2) "아직 학습 안 한" 데이터
        if row['llm_label'] == '유출' and unique_id not in trained_set:
            new_data_to_train.append(row)
            new_ids_to_log.append(unique_id)

    if not new_data_to_train:
        print("✅ 새로 학습할 '유출' 데이터가 없습니다. (모두 이전에 학습 완료)")
        return

    # 4. (시뮬레이션) 실제 학습 시작
    print(f"🔥 총 {len(new_data_to_train)}개의 '새로운 유출' 샘플로 뇌를 재학습(Fine-Tuning)합니다...")
    print("(실제 환경에서는 이 과정이 GPU로 몇 분/몇 시간이 걸릴 수 있습니다)")
    print("...")
    
    # (GPU가 일하는 척 30초간 대기)
    time.sleep(30) 
    
    print("...")
    print("✅ 재학습 완료!")

    # 5. '경력직' 뇌 저장 (시뮬레이션)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # "마지막 학습 시간" 기록 남기기
    with open(LAST_TRAINED_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Last trained at: {datetime.datetime.now()}")
    
    # (✨ 핵심) "학습 완료"된 ID들을 로그에 기록 (중복 학습 방지)
    for unique_id in new_ids_to_log:
        save_trained_log(unique_id)
        
    print(f"💾 '경력직' 뇌를 {MODEL_PATH}에 저장하고, {len(new_ids_to_log)}건을 '학습 완료' 처리했습니다.")
    print("🤖 3. '학습기' 봇(Trainer) 작동 완료.")

if __name__ == "__main__":
    main()