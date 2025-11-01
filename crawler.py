# 🕵️ (봇 1) '신입' 봇. '의심' 내역 수집 -> detected_leaks.csv
# ----------------------------------------------------
# (✨ 최종 로직)
# 1. 텍스트/이미지/GitHub에서 '의심' PII를 1차 수집합니다.
# 2. 'detected_leaks.csv' (In_1)와 'feedback_data.csv' (In_2)를 모두 확인합니다.
# 3. 두 곳 어디에도 없는 "진짜 새로운" 항목만 'detected_leaks.csv' (In_1)에 추가합니다.
# ----------------------------------------------------

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from urllib.parse import urljoin

# 우리 헬퍼 및 설정 파일 임포트
import config
import ocr_helper 

# --- 1. 설정값 ---
BASE_PATH = "/root/PII-Guardian" # (중요) deploy.yml의 DEPLOY_DIR과 일치
CSV_FILE = os.path.join(BASE_PATH, 'detected_leaks.csv')
FEEDBACK_FILE = os.path.join(BASE_PATH, 'feedback_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'my-ner-model')
BASE_MODEL = 'klue/roberta-base' # 🧠 기본 뇌 (Hugging Face)

REGEX_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b010[-.\s]?\d{4}[-.\s]?\d{4}\b',
}

# (✨ 핵심 수정) 
# 웹 URL 대신, 서버 로컬의 'test_site' 폴더를 직접 읽습니다.
TEST_FILES = [
    os.path.join(BASE_PATH, 'test_site/index.html'),
    os.path.join(BASE_PATH, 'test_site/page_with_image.html')
]

# (✨ 주석 처리) 
# GITHUB_QUERIES = [
#     '"ncp_api_key"',
#     '"IM뱅크" "비밀번호"',
# ]

# --- 2. 봇의 '뇌' (AI 모델) 로드 (✨ 최종 수정) ---
def load_ner_pipeline():
    """봇의 '뇌'(NER 모델)를 로드합니다."""
    
    # deploy.yml이 생성한 토큰 '파일'을 직접 읽어서 사용합니다.
    token_file_path = "/root/.cache/huggingface/token"
    hf_token = None
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, 'r') as f:
                hf_token = f.read().strip()
            if hf_token:
                 print("✅ Hugging Face 토큰 파일을 성공적으로 읽었습니다.")
            else:
                 print("⚠️ [경고] /root/.cache/huggingface/token 파일이 비어있습니다.")
                 print("⚠️ GitHub Secret 'HF_TOKEN'에 값이 올바르게 입력되었는지 확인하세요!")
        except Exception as e:
            print(f"⚠️ [경고] Hugging Face 토큰 파일 읽기 실패: {e}")
    else:
        print("⚠️ [경고] Hugging Face 토큰 파일(/root/.cache/huggingface/token)을 찾을 수 없습니다.")

    # (차선책) 파일이 없을 경우, config.py에서도 시도
    if not hf_token:
        hf_token = getattr(config, 'HF_TOKEN', None)
        if hf_token:
             print("✅ config.py에서 HF_TOKEN을 로드했습니다.")
    
    if not hf_token:
        print("❌ [치명적 오류] Hugging Face 토큰을 찾을 수 없어 모델을 로드할 수 없습니다.")
        return None 

    try:
        # 1순위: 우리가 학습시킨 '경력직' 뇌(my-ner-model)를 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=hf_token)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, token=hf_token)
        print(f"✅ '경력직' AI 뇌({MODEL_PATH}) 로드 성공!")
        
    # (✨ 핵심 수정) 
    # except OSError: -> except Exception:
    # '경력직' 뇌 로드에 "어떤 이유로든" (OSError, ValueError 등) 실패하면
    # '신입' 뇌를 로드하도록 합니다.
    except Exception as e: 
        print(f"⚠️ '경력직' AI 뇌({MODEL_PATH}) 로드 실패. 원인: {e}")
        print(f"➡️ '신입' 뇌({BASE_MODEL})를 로드합니다.")
        
        # 2순위: '신입' 뇌(BASE_MODEL)를 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
            model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, token=hf_token)
        except Exception as e2:
            print(f"❌ [치명적 오류] '신입' 뇌({BASE_MODEL}) 로드에도 실패했습니다: {e2}")
            return None
        
    # AI 모델을 사용하기 쉽게 '파이프라인'으로 만듦
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=-1, aggregation_strategy="simple")
    return ner_pipeline

# --- 3. 유출 탐지 함수 (텍스트용) ---
def find_leaks_in_text(text, ner_pipeline):
    """주어진 텍스트에서 RegEx와 NER로 PII를 찾습니다."""
    leaks = []
    
    context_preview = text.strip().replace('\n', ' ').replace('\r', ' ')[0:300]
    
    # 1. 정규식(RegEx)으로 먼저 탐지
    for pii_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            leaks.append({
                'type': pii_type,
                'content': match.group(0),
                'context': context_preview
            })
            
    # 2. AI(NER)로 추가 탐지
    try:
        ner_results = ner_pipeline(text[:512]) 
        for entity in ner_results:
            # klue/roberta-base는 'PS'(사람), 'LC'(장소), 'OG'(기관) 등을 탐지
            if entity['entity_group'] in ['PS', 'LC', 'OG']:
                leak_type = entity['entity_group']
                # 'PS' -> 'PERSON (AI)'처럼 좀 더 친절하게 변경
                if leak_type == 'PS': leak_type = 'PERSON (AI)'
                if leak_type == 'LC': leak_type = 'LOCATION (AI)'
                if leak_type == 'OG': leak_type = 'ORGANIZATION (AI)'
                
                leaks.append({
                    'type': leak_type,
                    'content': entity['word'],
                    'context': context_preview
                })
    except Exception as e:
        print(f"❌ [AI 분석 에러] {e}")
            
    return leaks

# --- 4. 크롤링 함수 (✨ 로컬 파일 읽기) ---
def crawl_local_file(file_path, ner_pipeline):
    """(기능 1) 하나의 '로컬 테스트 파일'을 읽습니다."""
    print(f"🕵️ [로컬 테스트] 파일 읽기 시작: {file_path}")
    leaks_found = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if not soup.body: return []
        page_text = soup.body.get_text(separator=' ')
        
        # 4-1. 텍스트
        leaks_found.extend(find_leaks_in_text(page_text, ner_pipeline))
        
        # 4-2. (✨ 주석 처리) OCR 기능
        # print("🖼️  이미지 스캔 기능을 주석 처리합니다.")
        
        return leaks_found
    except FileNotFoundError:
        print(f"❌ [에러] {file_path} 파일을 찾을 수 없습니다.")
        return []
    except Exception as e:
        print(f"❌ [에러] {file_path} 파일 처리 실패: {e}")
        return []

# --- 5. (✨ 주석 처리) 깃허브 검색 함수 ---
# def search_github_api(query, ner_pipeline): ...

# --- 6. CSV 저장 함수 ---
def get_existing_keys(file_path):
    """CSV 파일에서 (content, url) 키 세트를 로드합니다."""
    if not os.path.exists(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return set()
        df['url'] = df['url'].fillna('N/A') # URL 없는 경우 대비
        return set(zip(df['content'], df['url']))
    except pd.errors.EmptyDataError:
        return set()
    except Exception as e:
        print(f"⚠️ {file_path} 로드 중 오류: {e}")
        return set()

def save_to_csv(all_leaks):
    """탐지된 모든 내역을 '의심' 목록(CSV)에 '추가'합니다."""
    if not all_leaks:
        return
            
    new_df = pd.DataFrame(all_leaks)
    new_df['url'] = new_df['url'].fillna('N/A')
    
    # (✨ 핵심 1) 이미 처리된 '정답' 목록(feedback)에 있는지 확인
    processed_keys = get_existing_keys(FEEDBACK_FILE)
    
    # (✨ 핵심 2) 이미 '의심' 목록(detected)에 있는지 확인
    pending_keys = get_existing_keys(CSV_FILE)
    
    # (✨ 핵심 3) 두 곳 모두에 없는 "진짜 새로운" 항목만 필터링
    all_known_keys = processed_keys.union(pending_keys)
    
    is_truly_new = new_df.apply(lambda row: (row['content'], row['url']) not in all_known_keys, axis=1)
    final_new_df = new_df[is_truly_new]
    
    if final_new_df.empty:
        print("✅ 새로 발견된 '의심' 내역이 없습니다. (모두 기존 목록에 존재)")
        return

    # (✨ 핵심 4) "진짜 새로운" 항목만 '의심' 목록(detected_leaks.csv)에 '추가'
    print(f"✨ {len(final_new_df)}건의 '진짜 신규' 내역을 {CSV_FILE}에 추가합니다.")
    final_new_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False, encoding='utf-8-sig')

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    print("🤖 1. '신입' 봇(Crawler) 작동 시작...")
    
    print("🧠 봇의 AI 뇌(NER 모델)를 로드하는 중...")
    ner_brain = load_ner_pipeline()
    
    if ner_brain is None:
        print("❌ AI 뇌 로드에 실패하여 '신입' 봇을 종료합니다.")
        exit() # 스크립트 종료
        
    print("🧠 AI 뇌 로드 완료.")
    
    total_leaks_found = []
    
    # (✨ 수정) 로컬 테스트 파일 읽기
    for file_path in TEST_FILES:
        leaks = crawl_local_file(file_path, ner_brain)
        for leak in leaks:
            leak['url'] = os.path.basename(file_path) # url 대신 파일명(index.html) 기록
            leak['repo'] = 'test-site'
        total_leaks_found.extend(leaks)
        
    # (✨ 주석 처리) GitHub API 검색
    # print("🛰️ [GitHub API] 검색을 시작합니다...")
            
    # 최종 결과 저장
    if total_leaks_found:
        save_to_csv(total_leaks_found)
    
    print("🤖 1. '신입' 봇(Crawler) 작동 완료.")