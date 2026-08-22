# PII-Guardian

웹 페이지를 긁어서 개인정보 노출을 찾는 봇입니다.
찾은 것을 사람이 아니라 LLM에게 채점시키고, 그 채점 결과로 모델을 다시 학습시킵니다.
네이버 클라우드 서버에서 매일 자동으로 한 바퀴 돕니다.

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/HyperCLOVA_X-03C75A?style=flat-square&logo=naver&logoColor=white"/>
  <img src="https://img.shields.io/badge/Naver_Cloud-03C75A?style=flat-square&logo=naver&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
</p>

<br/>

## 1. 풀려던 문제

개인정보 탐지 모델을 학습시키려면 정답이 달린 문장이 수만 개 필요합니다.
그걸 손으로 달면 혼자서는 못 끝냅니다.

그래서 정답 다는 일을 LLM에게 맡겼습니다. 이게 이 프로젝트의 전부입니다.

<br/>

## 2. 한 바퀴 도는 구조

```
crawler.py       웹 수집. NER 모델 + 정규식으로 의심 후보를 뽑는다
     │            → detected_leaks.csv
     ▼
autolabeler.py   HyperCLOVA X 에게 "이게 진짜 유출인가" 를 묻는다
     │            → feedback_data.csv  (정답 라벨)
     ▼
train.py         유출로 판정된 것만 모아 NER 모델을 다시 학습시킨다
     │            → my-ner-model
     └──────────▶ 다음 crawler 실행부터 새 모델이 쓰인다

dashboard.py     세 단계의 현황과 로그를 한 화면에서 본다 (Streamlit)
```

서버에서 crontab 으로 돌립니다. 수집은 매시 정각, 라벨링은 01:00, 재학습은 02:00 입니다.

<br/>

## 3. 각 단계에서 한 판단

### 수집: 모델이 없어도 죽지 않게

`crawler.py` 는 `my-ner-model` 을 먼저 찾고, 없으면 베이스 모델로 떨어집니다.
처음 배포했을 때는 학습된 모델이 아직 없기 때문입니다.

```python
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer,
                        device=-1, aggregation_strategy="simple")
```

`aggregation_strategy="simple"` 을 쓴 이유는, 토큰 단위로 쪼개진 결과를 그대로 두면
`010`, `-`, `1234` 가 따로 나와서 사람이 못 읽기 때문입니다.
`device=-1` 은 CPU 입니다. 학습 서버가 GPU 없는 사양이라 추론은 CPU 로 돌립니다.

정규식은 모델과 같이 씁니다. 주민번호나 카드번호처럼 형식이 확실한 건 정규식이 더 정확하고,
문맥이 필요한 건 모델이 낫습니다.

### 라벨링: 온도를 낮춰서 답을 고정

`llm_helper.py` 가 HyperCLOVA X `HCX-005` 를 호출합니다.

```python
"temperature": 0.1
```

같은 문장을 두 번 물었을 때 답이 달라지면 학습 데이터가 오염됩니다.
그래서 온도를 낮춰서 판단을 고정했습니다.

### 재학습: 유출로 확정된 것만, 그리고 한 번만

`train.py` 는 BIO 태깅으로 라벨을 만듭니다.

```python
label_list = ['O', 'B-PII', 'I-PII']
```

개체의 첫 토큰은 `B-PII`, 이어지는 토큰은 `I-PII`, 나머지는 `O` 입니다.
서브워드로 쪼개진 조각과 특수 토큰은 `-100` 을 넣어서 손실 계산에서 뺍니다.

학습에 넣는 조건은 두 개입니다.

1. LLM 판정이 `유출` 인 것만
2. 이미 학습한 항목은 `trained_set` 으로 걸러서 다시 넣지 않는다

두 번째가 없으면 같은 데이터가 매일 다시 학습돼서 그쪽으로 모델이 치우칩니다.

<br/>

## 4. 채점하려고 사이트를 직접 만들었다

모델이 잘 찾는지 재려면 **정답을 아는 페이지**가 있어야 합니다.
실제 사이트를 긁어서 재면 정답을 모르니 재현율을 구할 수 없습니다.

그래서 `test_site/` 에 개인정보가 일부러 박힌 페이지를 만들었습니다.
`Faker` 한국어 로케일로 생성하고, 다음 아홉 가지를 심습니다.

주민등록번호, 전화번호, 이메일, 주소, 카드번호, 계좌번호, 여권번호, 운전면허번호, 외국인등록번호

여기에 두 가지를 더 넣었습니다.

- **난독화**: base64 로 감싸거나 `010-일이삼사-오육칠팔` 처럼 한글을 섞거나
  `park_suk (at) company [dot] co [dot] kr` 처럼 봇을 피하는 형태
- **오탐 유도**: 포트 번호 `8080`, 버전 `4.1.0-prod` 처럼 숫자지만 개인정보가 아닌 것

오탐 유도를 넣은 이유는, 잘 찾는 것만큼 **안 잡아야 할 걸 안 잡는 것**도 성능이기 때문입니다.

이 서버는 따로 띄워서 다른 에이전트 프로젝트의 테스트 대상으로도 쓸 수 있습니다.

<br/>

## 5. 네이버 클라우드 배포

GitHub Actions 워크플로 두 개로 돌립니다.

### `deploy.yml` : main 에 푸시하면 배포

`appleboy/ssh-action` 으로 NCP VM 에 접속해서 코드를 받고, venv 를 만들고,
crontab 세 줄을 등록하고, 대시보드 서비스를 재시작합니다.

접속 정보와 키는 전부 GitHub Secrets 입니다.
`NCP_HOST`, `NCP_USERNAME`, `NCP_SSH_KEY`, `HF_TOKEN` 을 씁니다.

**여기서 제일 오래 걸린 건 Chrome 설치였습니다.**

크롤링에 Chrome 과 ChromeDriver 가 필요한데, apt 로 설치하려니 GPG 키 검증에서 계속 막혔습니다.
저장소 키를 추가하는 방법도 서버 환경에 따라 되다 안 되다 했습니다.

결국 apt 를 안 쓰기로 하고 이렇게 바꿨습니다.

```bash
LATEST_STABLE_JSON=$(curl -s "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json")
CHROME_URL=$(echo $LATEST_STABLE_JSON | jq -r '.channels.Stable.downloads.chrome[] | select(.platform=="linux64") | .url')
DRIVER_URL=$(echo $LATEST_STABLE_JSON | jq -r '.channels.Stable.downloads.chromedriver[] | select(.platform=="linux64") | .url')
```

Chrome for Testing 이 공개하는 JSON 에서 최신 안정 버전 URL 을 뽑아 `wget` 으로 받고
`/opt` 에 풀어서 `/usr/bin` 으로 심볼릭 링크를 겁니다.

이렇게 하면 GPG 도 저장소 등록도 필요 없고, **Chrome 과 ChromeDriver 버전이 항상 맞습니다.**
따로 받으면 버전이 어긋나서 드라이버가 안 붙는 일이 자주 생기는데 그것도 같이 해결됐습니다.

### `wake-up-server.yml` : 필요할 때만 서버를 켠다

학습 서버를 24시간 켜 두면 요금이 계속 나갑니다.
그래서 평소엔 인스턴스를 내려 두고 매일 한국시간 00:10 에만 깨웁니다.

NCP API 는 요청마다 HMAC-SHA256 서명이 필요합니다.

```bash
SIG_MESSAGE="$METHOD $API_PATH\n$TIMESTAMP\n$ACCESS_KEY"
SIGNATURE=$(echo -n "$SIG_MESSAGE" | openssl dgst -sha256 -hmac "$SECRET_KEY" -binary | base64)
curl -s -X GET \
  -H "x-ncp-apigw-timestamp:$TIMESTAMP" \
  -H "x-ncp-iam-access-key:$ACCESS_KEY" \
  -H "x-ncp-apigw-signature-v2:$SIGNATURE" \
  "$API_URL/vserver/v2/startServerInstances?serverInstanceNoList.1=$INSTANCE_NO"
```

서명 문자열에 개행이 들어가서 `echo -n` 과 `printf` 중 무엇을 쓰느냐로 서명이 달라집니다.
여기서 한참 401 을 받았습니다.

**서버를 켜는 것만큼 끄는 것도 운영이라는 걸 여기서 배웠습니다.**

<br/>

## 6. 구조

```
crawler.py                    264줄  수집과 1차 탐지
autolabeler.py                100줄  LLM 라벨링
train.py                      224줄  NER 파인튜닝
dashboard.py                  181줄  Streamlit 관제 화면
llm_helper.py                  99줄  HyperCLOVA X 호출
ocr_helper.py                  53줄  이미지 속 텍스트 추출
test_site/
├── app.py                     39줄  테스트 서버
├── generate_dataset_v3.py    198줄  PII 9종 + 난독화 + 오탐 생성기
├── index.html
├── page_with_image.html
└── personal_info.png                이미지에 박힌 개인정보 케이스
.github/workflows/
├── deploy.yml                       NCP SSH 배포
└── wake-up-server.yml               NCP API 로 인스턴스 기동
```

<br/>

## 7. 실행

API 키는 `config.py` 에 두고 `.gitignore` 에 넣었습니다. 저장소에 키가 없습니다.

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

```bash
cd test_site && python3 app.py          # 테스트 서버 :5000
```

```bash
streamlit run dashboard.py              # 관제 화면 :8501
```

```bash
python3 crawler.py && python3 autolabeler.py && python3 train.py
```

<br/>

## 8. 한계

- 성능을 숫자로 못 냈습니다. 테스트 사이트는 만들었는데 정답셋 대비 재현율과 정밀도를 정리해 두지 않았습니다
- LLM 판정을 정답으로 씁니다. LLM 이 틀리면 그 오류가 그대로 모델에 들어갑니다. 표본 검수 절차가 없습니다
- `유출` 로 판정된 것만 학습해서 음성 예시가 부족합니다. 오탐이 늘어날 수 있는 구조입니다
- OCR 은 붙여만 뒀고 정확도를 재지 않았습니다
- 테스트가 없습니다

<br/>

MIT License
