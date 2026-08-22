# 🕵️ (봇 1) '신입' 봇. '의심' 내역 수집 -> detected_leaks.csv
# (v3.1 - Selenium 제거, Requests 복귀, Raw URL 스캔, 문맥(Context) 로직 수정)

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import sys
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging
# (✨ Selenium 관련 모듈 모두 삭제)

# 우리 헬퍼 및 설정 파일 임포트
import config

# --- 1. 설정값 ---
BASE_PATH = "/root/PII-Guardian"
LOG_FILE = os.path.join(BASE_PATH, 'crawler.log')

# (✨ 로그 중복 제거)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

CSV_FILE = os.path.join(BASE_PATH, 'detected_leaks.csv')
FEEDBACK_FILE = os.path.join(BASE_PATH, 'feedback_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'my-ner-model')
BASE_MODEL = 'klue/roberta-base' 

# (✨ v2.21 정규식)
REGEX_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b\(?(010)\)?[-.)\s]*\d{3,4}[-.\s]*\d{4}\b',
    'RRN': r'\b\d{6}[- ]*[1-4]\d{6}\b', 
    'CREDIT_CARD': r'\b\d{4}[- ]*\d{4}[- ]*\d{4}[- ]*\d{4}\b', 
    'ACCOUNT_NUM': r'\b\d{3}[- ]*\d{2,6}[- ]*\d{2,7}\b', 
    'API_KEY': r'\b(sk|pk|im-key-prod)-[a-zA-Z0-9_,-]{20,}\b',
    'INTERNAL_IP': r'\b(192\.168\.\d{1,3}\.\d{1,3})\b|\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
    'PHONE_GENERAL': r'\b\(?(0[2-9][0-9]?)\)?[-.)\s]*\d{3,4}[-.\s]*\d{4}\b|\b(15\d{2}|16\d{2})[-.\s]*\d{4}\b'
}

# (✨✨✨ 핵심 수정: GitHub 'Raw' URL로 변경 ✨✨✨)
# (Selenium이 필요 없는 '진짜' 원본 파일 주소)
CRAWL_URLS = [
    "http://10.0.1.6:5000/"]

# (✨ Selenium 드라이버 설정 함수 삭제)

# --- 2. 봇의 '뇌' (AI 모델) 로드 ---
def load_ner_pipeline():
    """봇의 '뇌'(NER 모델)를 로드합니다."""
    token_file_path = "/root/.cache/huggingface/token"
    hf_token = None
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, 'r') as f:
                hf_token = f.read().strip()
            if hf_token:
                 logging.info("✅ Hugging Face 토큰 파일을 성공적으로 읽었습니다.")
            else:
                 logging.warning("⚠️ [경고] /root/.cache/huggingface/token 파일이 비어있습니다.")
        except Exception as e:
            logging.warning(f"⚠️ [경고] Hugging Face 토큰 파일 읽기 실패: {e}")
    else:
        logging.warning("⚠️ [경고] Hugging Face 토큰 파일(/root/.cache/huggingface/token)을 찾을 수 없습니다.")

    if not hf_token:
        hf_token = getattr(config, 'HF_TOKEN', None)
        if hf_token:
             logging.info("✅ config.py에서 HF_TOKEN을 로드했습니다.")
    
    if not hf_token:
        logging.error("❌ [치명적 오류] Hugging Face 토큰을 찾을 수 없어 모델을 로드할 수 없습니다.")
        return None 

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=hf_token)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, token=hf_token)
        logging.info(f"✅ '경력직' AI 뇌({MODEL_PATH}) 로드 성공!")
    except Exception as e: 
        logging.warning(f"⚠️ '경력직' AI 뇌({MODEL_PATH}) 로드 실패. 원인: {e}")
        logging.info(f"➡️ '신입' 뇌({BASE_MODEL})를 로드합니다.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
            model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, token=hf_token)
        except Exception as e2:
            logging.error(f"❌ [치명적 오류] '신입' 뇌({BASE_MODEL}) 로드에도 실패했습니다: {e2}")
            return None
        
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=-1, aggregation_strategy="simple")
    return ner_pipeline

# --- 3. (✨✨✨ 핵심 수정 v3.1: '문맥' 로직 수정 ✨✨✨) ---
def find_leaks_in_text(text, ner_pipeline):
    """주어진 텍스트에서 RegEx와 NER로 PII를 찾습니다."""
    leaks = []
    if not text: 
        return leaks
        
    # (✨ 수정) 페이지 전체 300자가 아닌, PII 주변의 문맥을 저장합니다.
    # context_preview = text.strip().replace('\n', ' ').replace('\r', ' ')[0:300] # (버그가 있던 코드 삭제)
    
    for pii_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            
            # (✨ 신규) PII를 중심으로 앞뒤 150자, 총 300자 내외의 문맥을 생성합니다.
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 150)
            context_preview = text[start:end].strip().replace('\n', ' ').replace('\r', ' ')
            
            is_duplicate = False
            for existing_leak in leaks:
                if existing_leak['content'] == match.group(0):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                leaks.append({
                    'type': pii_type.replace('_GENERAL', ''),
                    'content': match.group(0),
                    'context': context_preview # (✨ 이제 올바른 문맥이 저장됨)
                })
            
    try:
        # (✨ 수정) 페이지 상단 512 토큰이 아닌, 텍스트 전체를 스캔합니다.
        ner_results = ner_pipeline(text) 
        
        for entity in ner_results:
            if entity['entity_group'] in ['PS', 'LC', 'OG', 'PII']: 
                
                # (✨ 신규) NER 결과에 대해서도 PII 중심의 문맥을 생성합니다.
                start = max(0, entity['start'] - 150)
                end = min(len(text), entity['end'] + 150)
                context_preview = text[start:end].strip().replace('\n', ' ').replace('\r', ' ')

                leak_type = entity['entity_group']
                if leak_type == 'PS': leak_type = 'PERSON (AI)'
                if leak_type == 'LC': leak_type = 'LOCATION (AI)'
                if leak_type == 'OG': leak_type = 'ORGANIZATION (AI)'
                if leak_type == 'PII': leak_type = 'PII (Custom AI)'
                
                leaks.append({
                    'type': leak_type,
                    'content': entity['word'],
                    'context': context_preview # (✨ 이제 올바른 문맥이 저장됨)
                })
    except Exception as e:
        logging.error(f"❌ [AI 분석 에러] {e}")
            
    return leaks

# --- 4. (✨ 수정) `requests` 기반 크롤링 함수 (OCR 비활성화) ---
def crawl_web_page(page_url, ner_pipeline):
    """(기능 1) `requests`로 정적 웹페이지를 크롤링합니다. (OCR은 비활성화)"""
    logging.info(f"🕵️ [Requests 크롤링] 시작: {page_url}")
    leaks_found = []
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Raw URL이므로 `response.text`가 순수 HTML입니다.
        html_content = response.text
        
        # (✨ 핵심) 4-1. HTML 주석()을 포함한 원본 텍스트 전체 스캔
        leaks_found.extend(find_leaks_in_text(html_content, ner_pipeline))
        
        # (✨ 핵심) 4-2. HTML 태그가 제거된, 눈에 보이는 텍스트 스캔
        soup = BeautifulSoup(html_content, 'html.parser')
        page_text = soup.get_text(separator=' ', strip=True)
        leaks_found.extend(find_leaks_in_text(page_text, ner_pipeline))

        # (OCR은 여전히 비활성화)
        
        return leaks_found
        
    except Exception as e:
        logging.error(f"❌ [Requests 크롤링 에러] {page_url} 처리 실패: {e}")
        return []

# --- 5. (주석 처리) 깃허브 검색 함수 ---
# (생략)

# --- 6. CSV 저장 함수 ---
def get_existing_keys(file_path):
    """CSV 파일에서 (content, url) 키 세트를 로드합니다."""
    if not os.path.exists(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return set()
        df['url'] = df['url'].fillna('N/A')
        return set(zip(df['content'], df['url']))
    except pd.errors.EmptyDataError:
        return set()
    except Exception as e:
        logging.warning(f"⚠️ {file_path} 로드 중 오류: {e}")
        return set()

def save_to_csv(all_leaks):
    """탐지된 모든 내역을 '의심' 목록(CSV)에 '추가'합니다."""
    if not all_leaks:
        return
            
    new_df = pd.DataFrame(all_leaks)
    new_df['url'] = new_df['url'].fillna('N.A')
    
    processed_keys = get_existing_keys(FEEDBACK_FILE)
    pending_keys = get_existing_keys(CSV_FILE)
    all_known_keys = processed_keys.union(pending_keys)
    
    is_truly_new = new_df.apply(lambda row: (row['content'], row['url']) not in all_known_keys, axis=1)
    
    # (✨ 수정) 중복 제거 (find_leaks_in_text가 2번 호출되므로)
    final_new_df = new_df[is_truly_new].drop_duplicates(subset=['content', 'url'])

    if final_new_df.empty:
        logging.info("✅ 새로 발견된 '의심' 내역이 없습니다. (모두 기존 목록에 존재)")
        return

    logging.info(f"✨ {len(final_new_df)}건의 '진짜 신규' 내역을 {CSV_FILE}에 추가합니다.")
    final_new_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False, encoding='utf-8-sig')

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    logging.info("🤖 1. '신입' 봇(Crawler) 작동 시작...")
    
    logging.info("🧠 봇의 AI 뇌(NER 모델)를 로드하는 중...")
    ner_brain = load_ner_pipeline() # <-- 변수명이 'ner_brain'
    if ner_brain is None:
        logging.error("❌ AI 뇌 로드에 실패하여 '신입' 봇을 종료합니다.")
        sys.exit(1)
    logging.info("🧠 AI 뇌 로드 완료.")

    # (✨ Selenium 드라이버 로드 코드 삭제)
    
    total_leaks_found = []
    
    # (✨ `requests` 기반 크롤링으로 변경)
    logging.info(f"🛰️ [Requests 크롤링] {len(CRAWL_URLS)}개의 URL을 스캔합니다. (OCR 비활성화)")
    for url in CRAWL_URLS:
        leaks = crawl_web_page(url, ner_brain) 
        for leak in leaks:
            leak['url'] = url 
            leak['repo'] = 'web-crawl'
        total_leaks_found.extend(leaks)
        time.sleep(1) # (사이트 부하 방지)

    # (✨ Selenium 드라이버 종료 코드 삭제)

    # (깃허브 API 검색은 여전히 주석 처리)
            
    # 최종 결과 저장 (로그 추가)
    if total_leaks_found:
        logging.info(f"✅ 총 {len(total_leaks_found)}개의 PII를 탐지했습니다. CSV 저장을 시작합니다.")
        save_to_csv(total_leaks_found)
    else:
        logging.info("✅ PII 탐지 결과: 0건. CSV 파일을 생성하지 않습니다.") 
    
    logging.info("🤖 1. '신입' 봇(Crawler) 작동 완료.")