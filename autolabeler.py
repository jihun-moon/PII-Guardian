# 🧑‍🏫 (봇 2) '전문가' 봇(LLM). 100% '자동' 정답 생성 -> feedback_data.csv
# ----------------------------------------------------
# 1. '신입' 봇이 모은 '의심' 목록(detected_leaks.csv)을 읽습니다.
# 2. HyperCLOVA(LLM)에게 '유출'/'공개'인지 물어봅니다.
# 3. LLM이 만든 '정답'을 '정답' 목록(feedback_data.csv)에 저장합니다.
# ----------------------------------------------------

import pandas as pd
import os
import llm_helper # (우리의 LLM 헬퍼 로드)
import time

DETECTED_FILE = 'detected_leaks.csv' # (입력) 신입 봇의 결과
FEEDBACK_FILE = 'feedback_data.csv' # (출력) 전문가 봇의 정답
PROCESSED_FILE = 'processed_detections.log' # (기록) 이미 처리한 항목

def load_processed():
    """이미 처리한 항목(중복 방지용)을 불러옵니다."""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
        # content + url을 합친 고유 ID를 set으로 저장
        return set(line.strip() for line in f)

def save_processed(unique_id):
    """처리 완료된 항목을 기록합니다."""
    with open(PROCESSED_FILE, 'a', encoding='utf-8') as f:
        f.write(unique_id + '\n')

def main():
    print("🤖 2. '전문가' 봇(AutoLabeler) 작동 시작...")
    
    # 1. '의심' 목록 파일(입력)이 있는지 확인
    if not os.path.exists(DETECTED_FILE):
        print("✅ '의심' 목록(detected_leaks.csv)이 없습니다. 작업을 종료합니다.")
        return

    # 2. '의심' 목록과 '이미 처리한' 목록을 로드
    try:
        detected_df = pd.read_csv(DETECTED_FILE)
    except pd.errors.EmptyDataError:
        print("✅ '의심' 목록(detected_leaks.csv)이 비어있습니다. 작업을 종료합니다.")
        return
        
    processed_set = load_processed()
    new_feedbacks = []
    
    print(f"총 {len(detected_df)}개의 '의심' 목록 발견. 이전에 처리하지 않은 항목을 검사합니다...")

    # 3. '의심' 목록을 하나씩 돌면서 '새로운' 항목만 처리
    for index, row in detected_df.iterrows():
        # 고유 ID 생성 (중복 처리 방지)
        unique_id = f"{row['content']}|{row['url']}"
        
        if unique_id not in processed_set:
            print(f"🧠 LLM(HyperCLOVA)에게 판단 요청: {row['content']}")
            
            # 4. LLM 헬퍼를 호출해 '유출'/'공개' 판단 요청
            try:
                # llm_helper.py의 함수 호출
                result = llm_helper.get_llm_judgment(row['context'], row['content'])
                
                # LLM의 답변을 '정답' 목록에 추가
                feedback = row.to_dict()
                feedback['llm_label'] = result.get('label', '오류') # "유출" or "공개"
                feedback['llm_reason'] = result.get('reason', 'N/A')
                new_feedbacks.append(feedback)
                
                # 처리 완료 기록
                save_processed(unique_id)
                
                # (API 과부하 방지를 위해 잠시 대기)
                time.sleep(1) 
                
            except Exception as e:
                print(f"❌ LLM 처리 중 에러: {e}")
        else:
            # print(f"이미 처리된 항목: {row['content']}") # (로그가 너무 많아질 수 있으니 주석 처리)
            pass

    # 5. 새로운 '정답'들을 '정답' 목록(feedback_data.csv)에 추가
    if new_feedbacks:
        print(f"✅ {len(new_feedbacks)}개의 새로운 '정답'을 생성했습니다. CSV 파일에 추가합니다.")
        new_feedback_df = pd.DataFrame(new_feedbacks)
        
        if os.path.exists(FEEDBACK_FILE):
            # 기존 파일에 이어서 쓰기 (header=False)
            new_feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            # 새 파일로 쓰기 (header=True)
            new_feedback_df.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
    else:
        print("✅ '신입' 봇이 찾은 모든 항목을 이미 '전문가' 봇이 처리했습니다.")

    print("🤖 2. '전문가' 봇(AutoLabeler) 작동 완료.")

if __name__ == "__main__":
    main()
