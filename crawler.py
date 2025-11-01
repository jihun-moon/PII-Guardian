# 🕵️ (봇 1) '신입' 봇. '의심' 내역 수집 -> detected_leaks.csv
# (v2.2 - 실제 웹사이트 크롤링 / OCR 기능은 비활성화)

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from urllib.parse import urljoin 
import logging

# 우리 헬퍼 및 설정 파일 임포트
import config
import ocr_helper # (✨ 임포트는 하되, 아래에서 사용 안 함)

# --- 1. 설정값 ---
BASE_PATH = "/root/PII-Guardian"
LOG_FILE = os.path.join(BASE_PATH, 'crawler.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])

CSV_FILE = os.path.join(BASE_PATH, 'detected_leaks.csv')
FEEDBACK_FILE = os.path.join(BASE_PATH, 'feedback_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'my-ner-model')
BASE_MODEL = 'klue/roberta-base' 

REGEX_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b010[-.\s]?\d{4}[-.\s]?\d{4}\b',
}

# --- 여기에 탐지하고 싶은 실제 웹사이트 주소를 넣으세요 ---
CRAWL_URLS = [
    "https://www.dcinside.com/"]

# --- 2. 봇의 '뇌' (AI 모델) 로드 ---
def load_ner_pipeline():
    """봇의 '뇌'(NER 모델)를 로드합니다."""
    # (내용 동일)
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

# --- 3. 유출 탐지 함수 (텍스트용) ---
def find_leaks_in_text(text, ner_pipeline):
    """주어진 텍스트에서 RegEx와 NER로 PII를 찾습니다."""
    # (내용 동일)
    leaks = []
    if not text: 
        return leaks
        
    context_preview = text.strip().replace('\n', ' ').replace('\r', ' ')[0:300]
    
    for pii_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            leaks.append({
                'type': pii_type,
                'content': match.group(0),
                'context': context_preview
            })
            
    try:
        ner_results = ner_pipeline(text[:512]) 
        for entity in ner_results:
            if entity['entity_group'] in ['PS', 'LC', 'OG']:
                leak_type = entity['entity_group']
                if leak_type == 'PS': leak_type = 'PERSON (AI)'
                if leak_type == 'LC': leak_type = 'LOCATION (AI)'
                if leak_type == 'OG': leak_type = 'ORGANIZATION (AI)'
                
                leaks.append({
                    'type': leak_type,
                    'content': entity['word'],
                    'context': context_preview
                })
    except Exception as e:
        logging.error(f"❌ [AI 분석 에러] {e}")
            
    return leaks

# --- 4. (✨ 수정) 실제 웹 크롤링 함수 (OCR 비활성화) ---
def crawl_web_page(page_url, ner_pipeline):
    """(기능 1) 하나의 '실제 웹페이지'를 크롤링합니다. (OCR은 비활성화)"""
    logging.info(f"🕵️ [웹 크롤링] 시작: {page_url}")
    leaks_found = []
    
    try:
        # 4-1. 웹페이지 다운로드
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        if not soup.body: 
            return []
            
        # 4-2. 페이지 텍스트 탐지
        page_text = soup.body.get_text(separator=' ')
        leaks_found.extend(find_leaks_in_text(page_text, ner_pipeline))
        
        # (✨✨✨ 핵심 수정 ✨✨✨)
        # --- 4-3. OCR 기능 주석 처리 ---
        # logging.info(f"🖼️  (OCR 기능 비활성화됨)")
        # images = soup.find_all('img')
        
        # for img in images:
        #     if not img.get('src'):
        #         continue
        #     image_url = urljoin(page_url, img['src'])
        #     if image_url.startswith('data:'):
        #         continue
        #     ocr_text = ocr_helper.get_ocr_text(image_url)
        #     if ocr_text:
        #         logging.info(f"  -> 👁️ OCR 스캔 성공: {image_url}")
        #         ocr_leaks = find_leaks_in_text(ocr_text, ner_pipeline)
        #         for leak in ocr_leaks:
        #             leak['type'] = f"{leak['type']} (Image)"
        #             leak['context'] = f"[이미지 스캔] {ocr_text[:200]}..."
        #         leaks_found.extend(ocr_leaks)
        # --- (주석 처리 끝) ---
        
        return leaks_found
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ [웹 크롤링 에러] {page_url} 다운로드 실패: {e}")
        return []
    except Exception as e:
        logging.error(f"❌ [웹 크롤링 에러] {page_url} 처리 실패: {e}")
        return []

# --- 5. (주석 처리) 깃허브 검색 함수 ---
# def search_github_api(query, ner_pipeline): ...

# --- 6. CSV 저장 함수 ---
def get_existing_keys(file_path):
    """CSV 파일에서 (content, url) 키 세트를 로드합니다."""
    # (내용 동일)
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
    # (내용 동일)
    if not all_leaks:
        return
            
    new_df = pd.DataFrame(all_leaks)
    new_df['url'] = new_df['url'].fillna('N/A')
    
    processed_keys = get_existing_keys(FEEDBACK_FILE)
    pending_keys = get_existing_keys(CSV_FILE)
    all_known_keys = processed_keys.union(pending_keys)
    
    is_truly_new = new_df.apply(lambda row: (row['content'], row['url']) not in all_known_keys, axis=1)
    final_new_df = new_df[is_truly_new]
    
    if final_new_df.empty:
        logging.info("✅ 새로 발견된 '의심' 내역이 없습니다. (모두 기존 목록에 존재)")
        return

    logging.info(f"✨ {len(final_new_df)}건의 '진짜 신규' 내역을 {CSV_FILE}에 추가합니다.")
    final_new_df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False, encoding='utf-8-sig')

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    logging.info("🤖 1. '신입' 봇(Crawler) 작동 시작...")
    
    logging.info("🧠 봇의 AI 뇌(NER 모델)를 로드하는 중...")
    ner_brain = load_ner_pipeline()
    
    if ner_brain is None:
        logging.error("❌ AI 뇌 로드에 실패하여 '신입' 봇을 종료합니다.")
        exit()
        
    logging.info("🧠 AI 뇌 로드 완료.")
    
    total_leaks_found = []
    
    # --- 실제 웹사이트 크롤링 시작 ---
    logging.info(f"🛰️ [실제 웹 크롤링] {len(CRAWL_URLS)}개의 URL을 스캔합니다. (OCR 비활성화)")
    for url in CRAWL_URLS:
        leaks = crawl_web_page(url, ner_brain)
        for leak in leaks:
            leak['url'] = url 
            leak['repo'] = 'web-crawl' 
        total_leaks_found.extend(leaks)
        time.sleep(1) 

    # (깃허브 API 검색은 여전히 주석 처리)
            
    # 최종 결과 저장
    if total_leaks_found:
        save_to_csv(total_leaks_found)
    
    logging.info("🤖 1. '신입' 봇(Crawler) 작동 완료.")