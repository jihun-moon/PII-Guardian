# 파일 이름: generate_dataset_v3.py (v2 대체)
# 위치: Pii-Guardian/test_site/
# (v3.0: PII 유형 추가, 난독화/오탐 패턴 대폭 강화)

import random
import base64
import urllib.parse
from faker import Faker

# 한국어 로케일로 Faker 인스턴스 생성
fake = Faker('ko_KR')

# --- 1. 유출(Leak) 데이터 생성 함수 (v3.0 신규 PII 추가) ---
def get_pii_phone(): return fake.phone_number()
def get_pii_ssn(): return fake.ssn() # 주민등록번호
def get_pii_email(): return fake.email()
def get_pii_address(): return fake.address()
def get_pii_card(): return fake.credit_card_number()
def get_pii_account():
    return f"{random.choice(['신한', '국민', '우리', 'DGB'])} {fake.numerify('###-######-##-###')}"
def get_pii_passport():
    """(신규) 여권번호 (M12345678)"""
    return f"M{fake.numerify('########')}"
def get_pii_driver_license():
    """(신규) 운전면허번호 (12-34-567890-11)"""
    return f"{fake.numerify('##-##-######-##')}"
def get_pii_alien_ssn():
    """(신규) 외국인등록번호 (900101-5123456)"""
    return f"{fake.numerify('######-[5-8]######')}"


# --- 2. 고난이도 유출 (v3.0 난독화 대폭 강화) ---

def get_pii_base64_hard():
    """(강화) JSON 객체 전체를 Base64로 인코딩"""
    pii_data = {
        "user_id": fake.user_name(),
        "contact_info": {
            "email": fake.email(),
            "phone": fake.phone_number()
        },
        "ssn": fake.ssn()
    }
    # JSON 문자열로 변환 후 Base64 인코딩
    json_string = str(pii_data)
    encoded_pii = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')
    return f"[DEBUG] UserData_Base64: {encoded_pii}"

def get_pii_url_encoded():
    """(신규) URL 인코딩된 PII (이메일, 주소 등)"""
    pii = f"mailto:{fake.email()}?subject=test&body=제 주소는 {fake.address()} 입니다."
    encoded_pii = urllib.parse.quote(pii)
    return f"Request Parameter: {encoded_pii}"

def get_pii_leetspeak():
    """(신규) Leet (1337) 변환 (카드번호, 이메일)"""
    email = fake.email().replace("o", "0").replace("i", "1").replace("e", "3").replace("a", "4")
    card = fake.credit_card_number().replace("0", "O").replace("1", "l").replace("5", "S")
    return random.choice([f"이멜: {email}", f"카두: {card}"])

def get_pii_homoglyph_hard():
    """(강화) 동형 문자 (키릴, 그리스 문자 등) 혼용"""
    email = fake.email().replace("o", "о").replace("a", "а").replace("c", "с") # 키릴
    phone = fake.phone_number().replace("0", "Ο").replace("1", "Ι") # 그리스
    return f"연락처(확인필요): {email} / {phone}"

def get_pii_split_hard():
    """(강화) PII를 3줄 이상으로 쪼개서 반환"""
    addr_parts = fake.address().split(' ')
    if len(addr_parts) < 3:
        addr_parts = ["서울시", "강남구", "테헤란로 123"]
    return (
        f"주소 1: {addr_parts[0]}",
        f"주소 2: {addr_parts[1]}",
        f"주소 3: {addr_parts[2]}"
    )

def get_pii_in_structure():
    """(신규) JSON, XML, YAML 구조체 안에 PII 숨기기"""
    email = fake.email()
    phone = fake.phone_number()
    choice = random.choice(['json', 'xml', 'yaml'])
    
    if choice == 'json':
        return f'{{"user": {{"id": 1, "contact": "{email}"}}}}'
    if choice == 'xml':
        return f'<user><id>1</id><contact>{phone}</contact></user>'
    if choice == 'yaml':
        return f'user:\n  id: 1\n  contact: {email}'
    return "" # 혹시 모를 경우 대비


# --- 3. 정상(Safe) 데이터 생성 함수 (v3.0 오탐/함정 강화) ---
def get_safe_id_hard():
    """(강화) 주민번호와 패턴이 '완벽히' 일치하는 주문번호"""
    return f"ORD-{fake.numerify(text='######-#######')}"

def get_safe_product_code_hard():
    """(강화) 카드번호와 패턴이 '완벽히' 일치하는 상품코드"""
    return f"PROD-{fake.numerify(text='####-####-####-####')}"

def get_safe_version_hard():
    """(강화) IP와 패턴이 '완벽히' 일치하는 버전명"""
    return f"v{fake.numerify(text='##.###.###.###')}"

def get_safe_uuid():
    """(신규) UUID (카드번호/키와 혼동 유발)"""
    return f"Session-ID: {fake.uuid4()}"

def get_safe_public_phone(): return random.choice(["1588-1234", "02-123-4567"])
def get_safe_public_email(): return "support@pii-guardian.com"
def get_safe_number(): return f"Build #{fake.numerify(text='#########')}"

# --- 4. 데이터와 컨텍스트(맥락) 무작위 조합 (v3.0 신규 컨텍스트 추가) ---

def generate_random_test_data(num_lines=200): # (기본 라인 수 200으로 증가)
    """지정된 줄 수만큼 무작위 PII HTML을 생성하여 '문자열'로 반환"""
    
    # (v3.0) 생성기 목록 대폭 강화
    pii_generators = [
        get_pii_phone, get_pii_ssn, get_pii_email, get_pii_address, get_pii_card, 
        get_pii_account, get_pii_passport, get_pii_driver_license, get_pii_alien_ssn,
        get_pii_base64_hard, get_pii_url_encoded, get_pii_leetspeak, 
        get_pii_homoglyph_hard, get_pii_split_hard, get_pii_in_structure
    ]
    safe_generators = [
        get_safe_id_hard, get_safe_product_code_hard, get_safe_version_hard,
        get_safe_uuid, get_safe_public_phone, get_safe_public_email, get_safe_number
    ]
    
    output_lines = []
    output_lines.append(f"""
    <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
    <title>랜덤 PII 테스트 (v7.0 - Hardened)</title>
    <style>
        body {{ font-family: 'Courier New', Courier, monospace; line-height: 1.6; padding: 20px; background: #f0f0f0; }}
        .warning {{ background-color: #fff0f0; border: 1px solid red; padding: 5px; margin: 3px; }}
        .safe {{ background-color: #f0f8ff; border: 1px solid #ccc; padding: 5px; margin: 3px; }}
        pre, code {{ background: #333; color: #f0f0f0; padding: 10px; border-radius: 5px; display: block; overflow-x: auto; }}
    </style>
    </head><body>
    <h1>실시간 랜덤 PII 테스트 (v7.0 - Hardened)</h1>
    <p>브라우저 또는 크롤러가 접속할 때마다 내용이 바뀝니다.</p>
    <hr>
    """)
    
    i = 0
    while i < num_lines:
        # (v3.0) 유출 확률 40%로 증가
        if random.random() < 0.40:
            generator = random.choice(pii_generators)
            pii_class = "warning"
            
            if generator == get_pii_split_hard:
                part1, part2, part3 = generator()
                output_lines.append(f'<p class="{pii_class}">Q: {part1}</p>')
                output_lines.append(f'<div class="{pii_class}">A: {part2}</div>')
                output_lines.append(f'<p class="{pii_class}">A2: {part3}</p>')
                i += 3 # 3줄 소모
                continue
            else:
                pii_data = generator()
        else:
            generator = random.choice(safe_generators)
            pii_data = generator()
            pii_class = "safe"

        # (v3.0) 컨텍스트(맥락) 추가
        context_choice = random.choice([
            'html_comment', 'p_tag', 'div_data', 'log', 'json_code',
            'xml_code', 'yaml_code', 'inline_script', 'img_alt_tag'
        ])
        
        line = ""
        # (v2.0 버그 수정 완료)
        if context_choice == 'html_comment':
            line = f''
        elif context_choice == 'p_tag':
            line = f'<p class="{pii_class}">문의 내용: {pii_data}</p>'
        elif context_choice == 'div_data':
            line = f'<div data-user-info="{pii_data}" class="{pii_class}">데이터 속성 (F12로 확인)</div>'
        elif context_choice == 'log':
            line = f'<pre class="{pii_class}">[INFO] {pii_data}</pre>'
        elif context_choice == 'json_code':
            line = f'<code class="{pii_class}">{{"key": "user_data", "value": "{pii_data}"}}</code>'
        elif context_choice == 'xml_code':
            line = f'<pre class="{pii_class}"><config><setting name="temp_key" value="{pii_data}" /></config></pre>'
        elif context_choice == 'yaml_code':
            line = f'<pre class="{pii_class}">config:\n  temp_key: {pii_data}</pre>'
        elif context_choice == 'inline_script':
            line = f'<script class="{pii_class}">var tempUserData = "{pii_data}";</script>'
        elif context_choice == 'img_alt_tag':
            line = f'<img src="fake.png" alt="QR Code Data: {pii_data}" class="{pii_class}" style="display:none;">'
            
        output_lines.append(line)
        i += 1

    output_lines.append("<hr></body></html>")
    return "\n".join(output_lines)