# 🤖 (AI) HyperCLOVA API 호출 도우미
# ----------------------------------------------------
# 'autolabeler.py'가 이 파일을 import하여 LLM의 판단을 받습니다.
# ----------------------------------------------------

import requests
import json
import config # (우리의 비밀 키 로드)

# HyperCLOVA X 모델에 보낼 시스템 프롬프트 (명령어)
SYSTEM_PROMPT = """
당신은 최고의 개인정보 보안 전문가입니다.
주어진 [문맥]에서 [탐지된 PII]가 발견되었습니다.
이것이 '의도치 않은 개인정보 유출'인지, 아니면 '공개적으로 제공된 연락처 정보'인지 판단하세요.

- '유출' (Leak): 비밀번호, API 키, 주민번호, 실수로 노출된 내부 이메일/전화번호 등
- '공개' (Public): 웹사이트 하단의 고객센터 이메일, 전화번호, 공식 주소 등

반드시 '유출' 또는 '공개' 둘 중 하나로만 답하고, 그 이유를 1줄로 설명하세요.
JSON 형식으로만 답하세요: {"label": "유출/공개", "reason": "이유"}
"""

def get_llm_judgment(context, pii_content):
    """
    HyperCLOVA X (CLOVA Studio) API를 호출하여
    탐지된 PII가 '유출'인지 '공개'인지 판단합니다.
    """
    
    # --- (✨ 수정된 부분) ---
    # 1. 사용할 모델 이름을 명시합니다. (v3 API와 호환되는 모델)
    MODEL_NAME = "HCX-005"
    
    # 2. config.py의 기본 URL에 '/v3/chat/completions/{MODEL_NAME}'을 결합합니다.
    API_URL = config.HCX_API_URL.rstrip('/') + f'/v3/chat-completions/{MODEL_NAME}'
    # --- (수정 끝) ---
    
    headers = {
        "Authorization": f"Bearer {config.HCX_API_KEY}", 
        "Content-Type": "application/json"
    }

    data = {
        # "model": "hcx-003", # <-- (✨ 수정) URL에 모델명이 포함되므로 body에서 제거
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"[문맥]: \"...{context}...\"\n[탐지된 PII]: \"{pii_content}\""
            }
        ],
        "response_format": {
            "type": "json_object" # JSON으로 답하도록 강제
        },
        "max_tokens": 100,
        "temperature": 0.1 # 일관된 답변을 위해 온도를 낮춤
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # v3 응답 구조 (choices[0].message.content)
        json_content = result['choices'][0]['message']['content']
        llm_answer = json.loads(json_content)
        
        return llm_answer # {"label": "...", "reason": "..."}
        
    except requests.exceptions.ReadTimeout:
        print("❌ [LLM API 에러] HyperCLOVA 타임아웃")
        return {"label": "오류", "reason": "타임아웃"}
    except Exception as e:
        print(f"❌ [LLM API 에러] {e}")
        # (디버깅을 위해 서버의 실제 응답을 출력)
        if 'response' in locals():
            print(f"    (응답: {response.text})")
        return {"label": "오류", "reason": str(e)}