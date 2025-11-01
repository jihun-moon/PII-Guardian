# 🕵️ (봇 1) '신입' 봇. '의심' 내역 수집 -> detected_leaks.csv
# ----------------------------------------------------
# 1. 텍스트/이미지(OCR)에서 '의심' PII를 1차 수집합니다.
# 2. (선택) GitHub API에서 '의심' PII를 1차 수집합니다.
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
# (데이터 저장 파일)
CSV_FILE = 'detected_leaks.csv'
FEEDBACK_FILE = 'feedback_data.csv' # 🧑‍🏫 (봇 2)의 '정답지'
# (NER 모델 경로)
MODEL_PATH = 'my-ner-model' # 🎓 (봇 3)이 훈련시킬 뇌
BASE_MODEL = 'klue/roberta-base-ner' # 🧠 기본 뇌 (Hugging Face)

# (1차 탐지용 정규식 패턴)
REGEX_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b010[-.\s]?\d{4}[-.\s]?\d{4}\b',
    # (패턴 추가 가능)
    # 'API_KEY': r'sk_[a-zA-Z0-9]{32,}' 
}

# (크롤링할 대상)
# 🚨 (수정) 사용자 이름을 'jihun0948'에서 'jihun-moon'으로 바로잡았습니다.
TEST_URLS = [
    'https://jihun-moon.github.io/PII-Guardian/test_site/index.html',
    'https://jihun-moon.github.io/PII-Guardian/test_site/page_with_image.html'
]

# (깃허브 검색어 - 주석 처리됨)
GITHUB_QUERIES = [
    '"ncp_api_key"',     # NCP API 키
    '"IM뱅크" "비밀번호"',
]

# --- 2. 봇의 '뇌' (AI 모델) 로드 (✨ 최종 수정) ---
def load_ner_pipeline():
    """봇의 '뇌'(NER 모델)를 로드합니다."""
    
    # (✨ 핵심 수정)
    # Crontab 환경 문제를 회피하기 위해, deploy.yml이 생성한
    # 토큰 '파일'을 직접 읽어서 사용합니다.
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
        except Exception as e:
            print(f"⚠️ [경고] Hugging Face 토큰 파일 읽기 실패: {e}")
    else:
        print("⚠️ [경고] Hugging Face 토큰 파일(/root/.cache/huggingface/token)을 찾을 수 없습니다.")

    # (차선책) 파일이 없을 경우, config.py에서도 시도
    if not hf_token:
        hf_token = getattr(config, 'HF_TOKEN', None)
        if hf_token:
             print("✅ config.py에서 HF_TOKEN을 로드했습니다.")

    try:
        # 1순위: 우리가 학습시킨 '경력직' 뇌(my-ner-model)를 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=hf_token)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, token=hf_token)
        print(f"✅ '경력직' AI 뇌({MODEL_PATH}) 로드 성공!")
    except OSError: 
        # 2순위: 1순위가 실패하면 '신입' 뇌(klue/roberta)를 로드
        print(f"⚠️ '경력직' AI 뇌({MODEL_PATH})를 찾을 수 없습니다. '신입' 뇌({BASE_MODEL})를 로드합니다.")
        
        # (✨ 핵심 3) '신입' 뇌 로드 시, 인증을 위해 토큰을 명시적으로 전달합니다.
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
        model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, token=hf_token)
        
    # AI 모델을 사용하기 쉽게 '파이프라인'으로 만듦
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=-1, aggregation_strategy="simple")
    return ner_pipeline

# --- 3. 유출 탐지 함수 (텍스트용) ---
def find_leaks_in_text(text, ner_pipeline):
# ... (이하 코드는 이전과 동일) ...
    """주어진 텍스트에서 RegEx와 NER로 PII를 찾습니다."""
    leaks = []
    
    # (문맥 저장을 위해 텍스트 길이 제한)
    context_preview = text.strip().replace('\n', ' ').replace('\r', ' ')[0:300]
    
    # 1. 정규식(RegEx)으로 먼저 탐지
    for pii_type, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            leaks.append({
                'type': pii_type,
                'content': match.group(0),
                'context': context_preview
            })
            
    # 2. AI(NER)로 추가 탐지 (예: 사람 이름)
    try:
        # (개선) 텍스트가 너무 길면 NER이 오류를 낼 수 있으므로 512자로 제한
        ner_results = ner_pipeline(text[:512]) 
        for entity in ner_results:
            # klue/roberta-base-ner는 'PS'(사람이름)을 탐지
            if entity['entity_group'] == 'PS':
                leaks.append({
                    'type': 'PERSON (AI)',
                    'content': entity['word'],
                    'context': context_preview
                })
    except Exception as e:
        print(f"❌ [AI 분석 에러] {e}")
            
    return leaks

# --- 4. 크롤링 함수 (테스트 사이트용) ---
def crawl_test_site(url, ner_pipeline):
    """(기능 1) 하나의 '테스트 URL'을 크롤링합니다."""
    print(f"🕵️ [테스트 사이트] 크롤링 시작: {url}")
    leaks_found = []
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8' 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # (개선) body가 없는 경우를 대비
        if not soup.body:
            return []
            
        page_text = soup.body.get_text(separator=' ')
        
        # 4-1. 텍스트에서 유출 탐지
        leaks_found.extend(find_leaks_in_text(page_text, ner_pipeline))
        
        # 4-2. 이미지(OCR)에서 유출 탐지
        images = soup.find_all('img')
        for img in images:
            try:
                img_url = img.get('src') # .get()으로 안전하게 접근
                if not img_url:
                    continue
                    
                # (상대 경로를 절대 경로로 변환)
                if not img_url.startswith('http'):
                    img_url = urljoin(url, img_url)
                
                print(f"🖼️  이미지 스캔 중... {img_url}")
                ocr_text = ocr_helper.get_ocr_text(img_url) # ocr_helper.py 호출
                
                if ocr_text:
                    image_leaks = find_leaks_in_text(ocr_text, ner_pipeline)
                    if image_leaks:
                        print(f"🚨 [OCR 탐지!] {img_url} 에서 {len(image_leaks)}건 발견!")
                        leaks_found.extend(image_leaks)
            except Exception as e:
                print(f"❌ [이미지 에러] {img.get('src')} 스캔 실패: {e}")

        return leaks_found
            
    except Exception as e:
        print(f"❌ [에러] {url} 크롤링 실패: {e}")
        return []

# --- 5. 깃허브 검색 함수 (주석 처리됨) ---
def search_github_api(query, ner_pipeline):
    """(기능 2) GitHub API로 '실제' 소스 코드를 검색합니다."""
    print(f"🛰️ [GitHub API] 검색 시작: {query}")
    
    API_URL = "https://api.github.com/search/code"
    headers = {
        "Authorization": f"token {getattr(config, 'GITHUB_TOKEN', '')}", 
        "Accept": "application/vnd.github.v3.text-match+json" 
    }
    params = {'q': query, 'sort': 'indexed', 'order': 'desc', 'per_page': 10} 
    
    total_leaks = []
    try:
        response = requests.get(API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status() 
        results = response.json()
        
        if 'items' not in results or not results['items']:
            print("✅ [GitHub API] 탐지된 내역 없음.")
            return []
            
        for item in results['items']:
            file_url = item['html_url']
            repo_name = item['repository']['full_name']
            
            code_context = ""
            if 'text_matches' in item and item['text_matches']:
                # (수정) text_matches는 여러 개일 수 있으므로 모두 합칩니다.
                code_context = " ... ".join([match['fragment'] for match in item['text_matches']])
            
            if code_context:
                leaks = find_leaks_in_text(code_context, ner_pipeline)
                for leak in leaks:
                    leak['url'] = file_url 
                    leak['repo'] = repo_name
                total_leaks.extend(leaks)
        
        if total_leaks:
            print(f"🚨 [GitHub 탐지!] 총 {len(total_leaks)}건 발견!")
        return total_leaks
        
    except Exception as e:
        print(f"❌ [GitHub API 에러] {e}")
        return []

# --- 6. CSV 저장 함수 (✨ 로직 대폭 개선) ---
def save_to_csv(all_leaks):
    """탐지된 모든 내역을 '의심' 목록(CSV)에 '추가'합니다."""
    if not all_leaks:
        return
            
    new_df = pd.DataFrame(all_leaks)
    
    # (✨ 개선 1) 이미 '정답지'에 있는 내역은 제외합니다.
    try:
        if os.path.exists(FEEDBACK_FILE):
            try:
                feedback_df = pd.read_csv(FEEDBACK_FILE)
            except pd.errors.EmptyDataError:
                feedback_df = pd.DataFrame(columns=['content', 'url']) # 빈 DataFrame

            if not feedback_df.empty:
                # '정답지'에 있는 (content, url) 쌍을 만듭니다.
                # (url이 없는 'test-site'의 경우를 대비해 fillna 사용)
                feedback_df['url'] = feedback_df['url'].fillna('test-site-url') # 임시 값
                new_df['url'] = new_df['url'].fillna('test-site-url') # 임시 값
                
                feedback_keys = set(zip(feedback_df['content'], feedback_df['url']))
                
                # (content, url)이 '정답지'에 없는 것만 필터링
                is_new = new_df.apply(lambda row: (row['content'], row['url']) not in feedback_keys, axis=1)
                new_df = new_df[is_new]
                
                if len(new_df) == 0:
                    print("✅ 새로 발견된 '의심' 내역이 없습니다. (모두 '정답지'에 이미 존재)")
                    return
                else:
                    print(f"✨ '정답지'와 비교 후, {len(new_df)}건의 '신규' 내역 발견!")

    except Exception as e:
        print(f"⚠️ '정답지'({FEEDBACK_FILE}) 비교 중 오류 발생: {e}")

    # (✨ 개선 2) '의심' 목록(detected_leaks.csv) 내의 중복도 제거합니다.
    if os.path.exists(CSV_FILE):
        try:
            existing_df = pd.read_csv(CSV_FILE)
            combined_df = pd.concat([existing_df, new_df])
            # 'content'와 'url'이 모두 똑같은 중복은 제거
            final_df = combined_df.drop_duplicates(subset=['content', 'url'])
        except pd.errors.EmptyDataError: # 파일이 비어있는 경우
            final_df = new_df
    else:
        final_df = new_df
        
    final_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"💾 '의심' 목록 저장 완료: {len(final_df)} 건")

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    print("🤖 1. '신입' 봇(Crawler) 작동 시작...")
    
    print("🧠 봇의 AI 뇌(NER 모델)를 로드하는 중...")
    ner_brain = load_ner_pipeline()
    print("🧠 AI 뇌 로드 완료.")
    
    total_leaks_found = []
    
    # (필수) 테스트 사이트 크롤링
    for url in TEST_URLS:
        leaks = crawl_test_site(url, ner_pipeline)
        for leak in leaks:
            leak['url'] = url 
            leak['repo'] = 'test-site'
        total_leaks_found.extend(leaks)
        
    # (선택) 실제 GitHub API 검색
    print("🛰️ [GitHub API] 검색을 시작합니다...")
    if not hasattr(config, 'GITHUB_TOKEN') or not config.GITHUB_TOKEN:
        print("⚠️ config.py에 GITHUB_TOKEN이 없습니다. GitHub 검색을 건너뜁니다.")
    else:
        for q in GITHUB_QUERIES:
            leaks = search_github_api(q, ner_brain)
            total_leaks_found.extend(leaks)
            time.sleep(5) # (중요) API 제한을 피하기 위해 5초간 휴식
            
    # 최종 결과 저장
    if total_leaks_found:
        save_to_csv(total_leaks_found)
    
    print("🤖 1. '신입' 봇(Crawler) 작동 완료.")

