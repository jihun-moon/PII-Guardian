# 파일 이름: generate_dataset_v2.py
# 위치: Pii-Guardian/test_site/

import random
import base64
from faker import Faker

# 한국어 로케일로 Faker 인스턴스 생성
fake = Faker('ko_KR')

# --- 1. 유출(Leak) 데이터 생성 함수 (기본) ---
def get_pii_phone(): return fake.phone_number()
def get_pii_ssn(): return fake.ssn()
def get_pii_email(): return fake.email()
def get_pii_address(): return fake.address()
def get_pii_card(): return fake.credit_card_number()
def get_pii_account():
    return f"{random.choice(['신한', '국민', '우리', 'DGB'])} {fake.numerify('###-######-##-###')}"

# --- 2. 고난이도 유출 (Hard Obfuscation) ---

def get_pii_base64():
    """Base64로 인코딩된 PII (매우 탐지하기 어려움)"""
    pii_list = [
        f"email: {fake.email()}",
        f"phone: {fake.phone_number()}",
        f"ssn: {fake.ssn()}"
    ]
    chosen_pii = random.choice(pii_list)
    # 문자열을 Base64로 인코딩
    encoded_pii = base64.b64encode(chosen_pii.encode('utf-8')).decode('utf-8')
    return f"[CONFIG] UserData: {encoded_pii}"

def get_obfuscated_email_hard():
    """HTML 엔티티, 보이지 않는 문자, 동형 문자 사용"""
    email = fake.email()
    email = email.replace("@", "&#64;").replace(".", "&#46;") # HTML 엔티티
    email = email[:3] + "\u200b" + email[3:] # 보이지 않는 문자
    email = email.replace("o", "о") # 동형 문자 (키릴 'о')
    return f"이메일: {email} (복사해서 사용하세요)"

def get_obfuscated_phone_hard():
    """한글, 특수문자, '노이즈' 혼합"""
    phone = fake.phone_number().split('-')
    patterns = [
        f"제 번호는 {phone[0]}... 아니 {phone[1]}... 아뇨 {phone[2]} 입니다.",
        f"연락처: {phone[0]} (일) {phone[1]} (이) {phone[2]}",
        f"HP: {phone[0]} / {phone[1]} / {phone[2]}"
    ]
    return random.choice(patterns)

def get_pii_split_hard():
    """PII를 두 줄로 쪼개서 반환 (LLM이 문맥을 이어야 함)"""
    ssn = fake.ssn().split('-')
    return (f"주민번호 앞자리는 {ssn[0]}", f"그리고 뒷자리는 {ssn[1]} 입니다.")

# --- 3. 정상(Safe) 데이터 생성 함수 (오탐 방지용) ---
def get_safe_id(): return f"ORD-{fake.numerify(text='######-#######')}"
def get_safe_public_phone(): return random.choice(["1588-1234", "02-123-4567"])
def get_safe_public_email(): return "support@pii-guardian.com"
def get_safe_number(): return f"Build #{fake.numerify(text='#########')}"

# --- 4. 데이터와 컨텍스트(맥락) 무작위 조합 (HTML 반환) ---

def generate_random_test_data(num_lines=150):
    """지정된 줄 수만큼 무작위 PII HTML을 생성하여 '문자열'로 반환"""
    
    pii_generators = [
        get_pii_phone, get_pii_ssn, get_pii_email, get_pii_address, get_pii_card, 
        get_pii_account, get_pii_base64, get_obfuscated_email_hard, 
        get_obfuscated_phone_hard, get_pii_split_hard
    ]
    safe_generators = [
        get_safe_id, get_safe_public_phone, get_safe_public_email, get_safe_number
    ]
    
    output_lines = []
    output_lines.append(f"""
    <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
    <title>랜덤 PII 테스트 (v6.0)</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        .warning {{ background-color: #fff0f0; border: 1px solid red; padding: 5px; margin: 3px; }}
        .safe {{ background-color: #f0f8ff; border: 1px solid #ccc; padding: 5px; margin: 3px; }}
        pre {{ background: #333; color: #f0f0f0; padding: 10px; border-radius: 5px; }}
    </style>
    </head><body>
    <h1>실시간 랜덤 PII 테스트 (v6.0)</h1>
    <p>브라우저 또는 크롤러가 접속할 때마다 내용이 바뀝니다.</p>
    <hr>
    """)
    
    i = 0
    while i < num_lines:
        if random.random() < 0.35: # 유출 데이터 생성 확률 35%
            generator = random.choice(pii_generators)
            pii_class = "warning"
            
            if generator == get_pii_split_hard:
                part1, part2 = generator()
                output_lines.append(f'<p class="{pii_class}">Q: {part1}</p>')
                output_lines.append(f'<div class="{pii_class}">A: {part2}</div>')
                i += 2 
                continue
            else:
                pii_data = generator()
        else:
            generator = random.choice(safe_generators)
            pii_data = generator()
            pii_class = "safe"

        context_choice = random.choice(['html_comment', 'p_tag', 'div_data', 'log', 'json_code'])
        
        line = ""
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
            
        output_lines.append(line)
        i += 1

    output_lines.append("<hr></body></html>")
    return "\n".join(output_lines)