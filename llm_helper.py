# 🤖 (AI) HyperCLOVA API 호출 도우미
# (v2.2 - Prompt Engineering + 로그 중복 제거)
# ----------------------------------------------------
# 'autolabeler.py'가 이 파일을 import하여 LLM의 판단을 받습니다.
# ----------------------------------------------------

import requests
import json
import config # (우리의 비밀 키 로드)
import logging # (✨ 신규)

# (✨ 신규) autolabeler와 같은 로거를 사용합니다.
logger = logging.getLogger(__name__)

# (✨✨✨ 핵심 수정: Prompt Engineering ✨✨✨)
# AI가 '신입' 봇의 오탐(False Positive)까지 걸러내도록 지침을 강화합니다.
SYSTEM_PROMPT = """
당신은 최고의 개인정보 보안 전문가입니다.
당신에게는 '신입' 봇이 1차로 탐지한 '의심' 목록이 전달됩니다.
'신입' 봇은 정규식(RegEx)에 기반하기에, **오탐(False Positive)을 자주 포함합니다.**

당신의 임무는 [문맥]을 비판적으로 검토하여, 이 탐지가 '진짜 유출'인지, 아니면 '오탐'이거나 '공개' 정보인지 판단하는 것입니다.

[분류 기준]
1.  **'유출' (Leak):**
    * 비밀번호, API 키, 주민번호, 여권번호, 카드번호, (공식 계정이 아닌) 개인 이메일/전화번호.
    * 문맥상 실수로 노출된 내부 정보 (예: 내부 IP, 직원 이름).

2.  **'공개' (Public):**
    * 웹사이트 하단의 고객센터 이메일, 대표 전화번호(1588-...), 공식 주소.
    * 'test@example.com', '010-0000-0000'처럼 명백한 **샘플 또는 테스트용 데이터.**
    * `100 101 102`처럼 **PII로 보기 어려운 단순 숫자/문자 나열.** (신입 봇의 오탐)

반드시 '유출' 또는 '공개' 둘 중 하나로만 답하고, 그 이유를 1줄로 설명하세요.
JSON 형식으로만 답하세요: {"label": "유출/공개", "reason": "이유"}
"""

def get_llm_judgment(context, pii_content):
    """
    HyperCLOVA X (CLOVA Studio) API를 호출하여
    탐지된 PII가 '유출'인지 '공개'인지 판단합니다.
    """
    
    MODEL_NAME = "HCX-005"
    API_URL = config.HCX_API_URL.rstrip('/') + f'/v3/chat-completions/{MODEL_NAME}'
    
    headers = {
        "Authorization": f"Bearer {config.HCX_API_KEY}", 
        "Content-Type": "application/json"
    }

    data = {
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
        
        # --- (✨ 핵심 수정) ---
        # v3 응답 구조가 'choices'가 아닌 'result' 키를 사용합니다.
        json_content = result['result']['message']['content']
        # --- (수정 끝) ---
        
        llm_answer = json.loads(json_content)
        
        return llm_answer # {"label": "...", "reason": "..."}
        
    except requests.exceptions.ReadTimeout:
        # (✨ 수정) print -> logger.error
        logger.error("❌ [LLM API 에러] HyperCLOVA 타임아웃")
        return {"label": "오류", "reason": "타임아웃"}
    except Exception as e:
        # (✨ 수정) print -> logger.error
        logger.error(f"❌ [LLM API 에러] {e}")
        if 'response' in locals():
            logger.error(f"    (응답: {response.text})")
        return {"label": "오류", "reason": str(e)}