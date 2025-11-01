# 🧑‍🏫 (봇 2) '전문가' 봇(LLM). 100% '자동' 정답 생성 -> feedback_data.csv
# ----------------------------------------------------
# (✨ 최종 로직: In/Outbox)
# 1. '의심' 목록 (detected_leaks.csv) [In_1]을 읽습니다.
# 2. "모든" 항목을 LLM에게 물어봅니다. (crawler.py가 이미 걸러줬기 때문)
# 3. '정답' 목록 (feedback_data.csv) [In_2]에 '추가'합니다.
# 4. 작업 완료 후 '의심' 목록 (detected_leaks.csv) [In_1]을 "삭제"합니다.
# ----------------------------------------------------

import pandas as pd
import os
import llm_helper # (우리의 LLM 헬퍼 로드)
import time

DETECTED_FILE = 'detected_leaks.csv' # (In_1) '받은 편지함'
FEEDBACK_FILE = 'feedback_data.csv' # (In_2) '보낸 편지함'

def main():
    print("🤖 2. '전문가' 봇(AutoLabeler) 작동 시작...")
    
    # 1. '의심' 목록(In_1) 파일이 있는지 확인
    if not os.path.exists(DETECTED_FILE):
        print("✅ '의심' 목록(In_1)이 없습니다. 작업을 종료합니다.")
        return

    # 2. '의심' 목록 로드
    try:
        detected_df = pd.read_csv(DETECTED_FILE)
        if detected_df.empty:
            print("✅ '의심' 목록(In_1)이 비어있습니다. 작업을 종료합니다.")
            # 비어있는 파일은 삭제
            os.remove(DETECTED_FILE)
            print(f"🗑️ 비어있는 {DETECTED_FILE} 파일을 정리했습니다.")
            return
            
    except pd.errors.EmptyDataError:
        print("✅ '의심' 목록(In_1)이 비어있습니다. 작업을 종료합니다.")
        os.remove(DETECTED_FILE)
        print(f"🗑️ 비어있는 {DETECTED_FILE} 파일을 정리했습니다.")
        return
    except Exception as e:
        print(f"❌ {DETECTED_FILE} 로드 중 에러: {e}. 작업을 중단합니다.")
        return
        
    print(f"총 {len(detected_df)}개의 새로운 '의심' 항목을 처리합니다...")
    new_feedbacks = []

    # 3. '의심' 목록을 "전부" 처리
    # (crawler.py가 이미 중복을 걸러줬으므로, 여기선 'processed.log'가 필요 없음)
    for index, row in detected_df.iterrows():
        print(f"🧠 LLM(HyperCLOVA)에게 판단 요청: {row['content']}")
        
        try:
            # llm_helper.py의 함수 호출
            result = llm_helper.get_llm_judgment(row['context'], row['content'])
            
            feedback = row.to_dict()
            feedback['llm_label'] = result.get('label', '오류') # "유출" or "공개"
            feedback['llm_reason'] = result.get('reason', 'N/A')
            new_feedbacks.append(feedback)
            
            # (API 과부하 방지를 위해 잠시 대기)
            time.sleep(1) 
            
        except Exception as e:
            print(f"❌ LLM 처리 중 에러: {e}")
            new_feedbacks.append({**row.to_dict(), 'llm_label': '오류', 'llm_reason': str(e)})

    # 4. 새로운 '정답'들을 '정답' 목록(In_2)에 '추가'
    if new_feedbacks:
        print(f"✅ {len(new_feedbacks)}개의 '정답'을 생성했습니다. {FEEDBACK_FILE}(In_2)에 추가합니다.")
        new_feedback_df = pd.DataFrame(new_feedbacks)
        
        new_feedback_df.to_csv(FEEDBACK_FILE, 
                               mode='a', 
                               header=not os.path.exists(FEEDBACK_FILE), 
                               index=False, 
                               encoding='utf-8-sig')
    else:
        print("⚠️ 처리할 항목이 있었으나, '정답'이 생성되지 않았습니다.")

    # 5. (✨ 핵심) 작업 완료 후 '의심' 목록(In_1)을 "삭제" (In/Outbox)
    try:
        os.remove(DETECTED_FILE)
        print(f"🗑️ 작업 완료. {DETECTED_FILE}(In_1)을 삭제했습니다.")
    except Exception as e:
        print(f"❌ {DETECTED_FILE} 삭제 중 에러: {e}")

    print("🤖 2. '전문가' 봇(AutoLabeler) 작동 완료.")

if __name__ == "__main__":
    main()