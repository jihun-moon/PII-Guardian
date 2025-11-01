# 파일 이름: app.py
# 위치: Pii-Guardian/test_site/

from flask import Flask, make_response
# (✨ 수정) v2 대신 v3에서 함수를 가져옴
from generate_dataset_v3 import generate_random_test_data

app = Flask(__name__)

@app.route('/')
def home():
    """
    메인 페이지('/')에 접속하면, 
    generate_random_test_data 함수를 호출하여 
    매번 새로운 200줄짜리 HTML 데이터를 생성합니다.
    """
    # num_lines 값을 조절하여 데이터 양을 늘릴 수 있습니다.
    html_content = generate_random_test_data(num_lines=200)
    
    response = make_response(html_content)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/text')
def text_only():
    """
    /text 페이지 (순수 텍스트만 반환 - 옵션)
    """
    html_content = generate_random_test_data(num_lines=200)
    
    import re
    text_content = re.sub(r'<[^>]+>', '\n', html_content)
    
    response = make_response(text_content)
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    return response

if __name__ == '__main__':
    # 서버 실행 (debug=True로 설정하면 코드 변경 시 자동 재시작)
    app.run(debug=True, host='0.0.0.0', port=5000)