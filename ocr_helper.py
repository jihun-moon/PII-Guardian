# (AI) CLOVA OCR API 호출 도우미
# ----------------------------------------------------
# 'crawler.py'가 이 파일을 import하여 이미지 속 글자를 읽습니다.
# ----------------------------------------------------

import requests
import json
import uuid
import time
import config # (우리의 비밀 키 로드)

def get_ocr_text(image_url):
    """
    CLOVA OCR API를 호출하여 이미지 URL에서 텍스트를 추출합니다.
    """
    headers = {
        "X-OCR-SECRET": config.OCR_SECRET_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "images": [
            {
                "format": "png", # (jpg, png 등 자동 감지되나, 확장자 명시 권장)
                "name": "temp_image",
                "data": None,
                "url": image_url
            }
        ],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(time.time() * 1000)
    }
    
    try:
        response = requests.post(config.OCR_API_URL, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status() # 200 OK가 아니면 에러
        
        result = response.json()
        
        # OCR 결과 텍스트를 하나로 합치기
        full_text = ""
        if 'images' in result and result['images']:
            for field in result['images'][0].get('fields', []):
                full_text += field.get('inferText', '') + " "
        
        return full_text
        
    except requests.exceptions.ReadTimeout:
        print(f"[OCR API 에러] {image_url} 타임아웃")
        return None
    except Exception as e:
        print(f"[OCR API 에러] {e}")
        return None