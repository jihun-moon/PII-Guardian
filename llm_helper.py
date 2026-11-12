# (AI) HyperCLOVA API 호출 도우미
# (v2.2 - Prompt Engineering + 로그 중복 제거)
# ----------------------------------------------------
# 'autolabeler.py'가 이 파일을 import하여 LLM의 판단을 받습니다.
# ----------------------------------------------------

import requests
import json
import config # (우리의 비밀 키 로드)
import logging # (신규)

# (신규) autolabeler와 같은 로거를 사용합니다.
logger = logging.getLogger(__name__)

# (핵심 수정 v2.2: Prompt Engineering 강화 )
# AI가 '신입' 봇의 오탐(False Positive)까지 걸러내도록 지침을 대폭 강화합니다.
SYSTEM_PROMPT = """
당신은 최고의 개인정보 보안 전문가(LLM)입니다.
당신에게는 '신입' 봇이 1차로 탐지한 '의심' 목록이 [탐지된 PII]와 [문맥] 세트로 전달됩니다.
'신입' 봇은 **오직 정규식(RegEx)만으로** 탐지하기 때문에, 문맥을 전혀 이해하지 못하며 **치명적인 오탐(False Positive)을 대량으로 포함합니다.**

당신의 임무는 [문맥]과 [탐지된 PII]를 비판적으로 검토하여, 이 탐지가 '진짜 유출'인지, 아니면 '신입' 봇의 '오탐'인지를 판단하는 것입니다.

[판단 로직]
1.  **[문맥]을 먼저 확인하십시오.** [문맥]이 'GitHub UI', 'Skip to content', 'Navigation Menu' 등과 같이 **웹사이트의 UI/메뉴 텍스트**인 경우, [탐지된 PII]가 실제 유출 정보(예: `im-key-prod-...`)가 아닌 한, **대부분 '공개' (오탐)일 가능성이 높습니다.**
2.  **[탐지된 PII]의 형태를 비판적으로 검토하십시오.**

[분류 기준]
1.  **'유출' (Leak):**
    * 비밀번호, API 키 (예: `im-key-prod-...`), 주민번호 (예: `901231-1234567`), 여권번호, 카드번호 (예: `4500-1234-...`).
    * 문맥상 명백히 **개인 식별 정보** (예: `sunsin_lee@naver.com`, `010-1111-2222`).
    * 내부 IP (예: `192.168.x.x`), 내부 서버 주소 (예: `jenkins.internal.dgb.co.kr`).

2.  **'공개' (Public) - (신입 봇의 오탐 포함):**
    * 웹사이트 하단의 고객센터 이메일(예: `support@...`, `recruit@...`), 대표 전화번호(예: `1588-XXXX`), 공식 주소.
    * `test@example.com`, `010-0000-0000`처럼 명백한 **샘플 또는 테스트용 데이터.**
    * **[가장 중요]** `100 101 102`, `124 125 126`, `123-45-67890`(사업자번호)처럼, **[탐지된 PII] 자체가 개인정보로 보기 어려운 단순 숫자/문자 나열**인 경우. (신입 봇이 계좌번호나 주민번호 정규식으로 오탐한 경우임).
    * [문맥]이 '이메일 정규식 예제:' 등이며 [탐지된 PII]가 `A-Za-z0-9` 같은 **코드/정규식의 일부**인 경우.

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
        
        # --- (핵심 수정) ---
        # v3 응답 구조가 'choices'가 아닌 'result' 키를 사용합니다.
        json_content = result['result']['message']['content']
        # --- (수정 끝) ---
        
        llm_answer = json.loads(json_content)
        
        return llm_answer # {"label": "...", "reason": "..."}
        
    except requests.exceptions.ReadTimeout:
        # (수정) print -> logger.error
        logger.error("[LLM API 에러] HyperCLOVA 타임아웃")
        return {"label": "오류", "reason": "타임아웃"}
    except Exception as e:
        # (수정) print -> logger.error
        logger.error(f"[LLM API 에러] {e}")
        if 'response' in locals():
            logger.error(f"    (응답: {response.text})")
        return {"label": "오류", "reason": str(e)}