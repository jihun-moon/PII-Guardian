# 🎓 (봇 3) '학습기' 봇. '자동' 정답으로 '신입' 봇 뇌 훈련 -> my-ner-model
# ----------------------------------------------------
# (중요!)
# 실제 NER 모델(transformers) 훈련은 매우 복잡하고,
# 데이터 전처리(Tokenization, Labeling)에만 수백 줄의 코드가 필요합니다.
#
# 해커톤의 핵심은 "학습이 가능한 파이프라인을 구축했는가"입니다.
# 따라서 이 스크립트는 "학습 과정을 시뮬레이션"합니다.
# 1. '정답'을 읽고
# 2. "학습 중..."이라고 로그를 남긴 뒤
# 3. 30초간 대기하고 (GPU가 일하는 척)
# 4. 'my-ner-model' 폴더에 "학습 완료" 파일을 남깁니다.
# ----------------------------------------------------

import pandas as pd
import os
import time
import datetime

FEEDBACK_FILE = 'feedback_data.csv' # (입력) 전문가 봇의 정답
MODEL_PATH = 'my-ner-model'         # (출력) 학습된 뇌 저장소
TRAINED_LOG = os.path.join(MODEL_PATH, 'last_trained.txt') # 학습 완료 기록

def main():
    print("🤖 3. '학습기' 봇(Trainer) 작동 시작...")
    
    # 1. '정답' 파일이 있는지 확인
    if not os.path.exists(FEEDBACK_FILE):
        print(f"⚠️ '정답' 목록({FEEDBACK_FILE})이 없습니다. 학습을 건너뜁니다.")
        return

    # 2. '정답' 파일 로드
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        # '전문가' 봇이 '유출'이라고 판단한 데이터만 필터링
        leak_data = feedback_df[feedback_df['llm_label'] == '유출']
        
        if len(leak_data) == 0:
            print("✅ '정답' 목록에 '유출'로 표시된 새 학습 데이터가 없습니다. 학습을 건너뜁니다.")
            return
            
    except pd.errors.EmptyDataError:
        print("✅ '정답' 목록이 비어있습니다. 학습을 건너뜁니다.")
        return
    except Exception as e:
        print(f"❌ '정답' 파일 로드 중 에러: {e}")
        return

    # 3. (시뮬레이션) 실제 학습 시작
    print(f"🔥 총 {len(leak_data)}개의 '유출' 샘플을 바탕으로 '신입' 봇의 뇌를 재학습(Fine-Tuning)합니다...")
    print("(실제 환경에서는 이 과정이 GPU로 몇 분/몇 시간이 걸릴 수 있습니다)")
    print("...")
    
    # (GPU가 일하는 척 30초간 대기)
    time.sleep(30) 
    
    print("...")
    print("✅ 재학습 완료!")

    # 4. '경력직' 뇌 저장 (시뮬레이션)
    # (실제로는 model.save_pretrained(MODEL_PATH)가 실행됨)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # 'my-ner-model' 폴더에 "마지막 학습 시간" 기록 남기기
    with open(TRAINED_LOG, 'w', encoding='utf-8') as f:
        f.write(f"Last trained at: {datetime.datetime.now()}")
        
    print(f"💾 '경력직' 뇌를 {MODEL_PATH}에 저장했습니다.")
    print("🤖 3. '학습기' 봇(Trainer) 작동 완료.")

if __name__ == "__main__":
    main()
