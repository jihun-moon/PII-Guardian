# 🎓 (봇 3) '학습기' 봇. '자동' 정답으로 '신입' 봇 뇌 훈련 -> my-ner-model
# (v2.4 - 체크포인트 2개 저장)
# ----------------------------------------------------
# 1. '정답' 목록 (feedback_data.csv) [In_2]를 읽습니다.
# 2. 'trained.log' (학습 기록)을 읽습니다.
# 3. [In_2]에만 있고 [학습 기록]에는 없는 "새로운 정답"만 학습합니다.
# 4. 학습 완료 후, "새로운 정답"의 ID를 'trained.log'에 '추가'합니다.
# 5. (✨ 신규) 재학습된 '경력직' 뇌를 'my-ner-model' 폴더에 저장합니다.
# ----------------------------------------------------

import pandas as pd
import os
import logging
import config # (✨ 신규) HF_TOKEN을 읽기 위해
from datasets import Dataset # (✨ 신규)
from transformers import ( # (✨ 신규)
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification # (✨ 패딩 오류 수정)
)

# --- 1. 설정값 ---
BASE_PATH = "/root/PII-Guardian" 
LOG_FILE = os.path.join(BASE_PATH, 'train.log')

# (✨ 로그 중복 제거)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

FEEDBACK_FILE = os.path.join(BASE_PATH, 'feedback_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'my-ner-model') # 🧠 '경력직' 뇌 저장 경로
TRAINED_LOG_FILE = os.path.join(BASE_PATH, 'trained.log')
BASE_MODEL = 'klue/roberta-base' # 🧠 '신입' 뇌 (기본 모델)

# (✨ 신규) NER 태그 정의 (IOB2 형식)
label_list = ['O', 'B-PII', 'I-PII']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


# --- 2. 로그 관리 함수 (기존과 동일) ---
def load_trained_log():
    """이미 학습한 항목(중복 학습 방지용)을 불러옵니다."""
    if not os.path.exists(TRAINED_LOG_FILE):
        return set()
    try:
        with open(TRAINED_LOG_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except Exception as e:
        logging.warning(f"⚠️ 'trained.log' 로드 실패: {e}")
        return set()

def save_trained_log(unique_id):
    """학습 완료된 항목을 기록합니다."""
    with open(TRAINED_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(unique_id + '\n')

# --- 3. (✨ 신규) 데이터 전처리 함수 ---
def preprocess_for_ner(new_data_df, tokenizer):
    """
    (context, content) 데이터를 NER 학습용 IOB2 태그로 변환합니다.
    """
    dataset_list = []
    
    for index, row in new_data_df.iterrows():
        context = str(row.get('context', ''))
        content = str(row.get('content', ''))

        if not context or not content:
            continue
            
        # 1. PII (content)가 문맥(context) 어디에 있는지 찾기
        start_idx = context.find(content)
        if start_idx == -1:
            logging.warning(f"⚠️ 학습 데이터 오류: PII '{content}'가 문맥 '{context[:50]}...'에 없습니다. 건너뜁니다.")
            continue
        end_idx = start_idx + len(content)

        # 2. 문맥(context)을 토크나이저로 분절
        tokenized_inputs = tokenizer(context, truncation=True, max_length=512, return_offsets_mapping=True)
        offsets = tokenized_inputs.pop("offset_mapping")
        
        # 3. 모든 토큰을 'O' (PII 아님)으로 초기화
        labels = [label2id['O']] * len(tokenized_inputs['input_ids'])
        is_b_token = True # 'B-PII' 태그를 붙였는지 확인

        # 4. 토큰의 위치(offset)와 PII 위치(start_idx, end_idx)를 비교
        for i, (offset_start, offset_end) in enumerate(offsets):
            # (예외 처리) [CLS], [SEP] 같은 특수 토큰
            if offset_start == 0 and offset_end == 0:
                labels[i] = -100 # loss 계산에서 제외
                continue

            # (핵심) 현재 토큰이 PII 범위 안에 포함되는지 확인
            if offset_start >= start_idx and offset_end <= end_idx:
                if is_b_token:
                    labels[i] = label2id['B-PII'] # 첫 토큰은 B-PII
                    is_b_token = False
                else:
                    labels[i] = label2id['I-PII'] # 나머지는 I-PII
            else:
                is_b_token = True # PII 범위를 벗어나면 B-PII 초기화
        
        tokenized_inputs['labels'] = labels
        dataset_list.append(tokenized_inputs)
        
    if not dataset_list:
        return None
        
    return Dataset.from_list(dataset_list)

# --- 4. 메인 실행 ---
def main():
    logging.info("🤖 3. '학습기' 봇(Trainer) 작동 시작...")
    
    # 1. '정답' 파일(In_2) 로드
    if not os.path.exists(FEEDBACK_FILE):
        logging.warning(f"⚠️ '정답' 목록({FEEDBACK_FILE})이 없습니다. 학습을 건너뜁니다.")
        return

    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        feedback_df['url'] = feedback_df['url'].fillna('N/A')
    except pd.errors.EmptyDataError:
        logging.info("✅ '정답' 목록이 비어있습니다. 학습을 건너뜁니다.")
        return
    except Exception as e:
        logging.error(f"❌ '정답' 파일 로드 중 에러: {e}")
        return
        
    # 2. '학습 기록' 로드 및 '새로운 학습 데이터' 필터링
    trained_set = load_trained_log()
    
    new_data_rows = []
    new_ids_to_log = []
    
    for index, row in feedback_df.iterrows():
        unique_id = f"{row['content']}|{row['url']}"
        # (조건 1) "유출" 라벨이고, (조건 2) "아직 학습 안 한" 데이터
        if row['llm_label'] == '유출' and unique_id not in trained_set:
            new_data_rows.append(row.to_dict())
            new_ids_to_log.append(unique_id)

    if not new_data_rows:
        logging.info("✅ 새로 학습할 '유출' 데이터가 없습니다. (모두 이전에 학습 완료)")
        return

    logging.info(f"🔥 총 {len(new_data_rows)}개의 '새로운 유출' 샘플로 뇌를 재학습(Fine-Tuning)합니다...")
    new_data_df = pd.DataFrame(new_data_rows)

    # 3. (✨ 신규) 모델과 토크나이저 로드
    HF_TOKEN = getattr(config, 'HF_TOKEN', None)
    if not HF_TOKEN:
        logging.error("❌ config.py에서 HF_TOKEN을 찾을 수 없습니다. 학습을 중단합니다.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        token=HF_TOKEN
    )
    
    # 4. (✨ 신규) 데이터 전처리
    logging.info("데이터 전처리(NER 태깅) 시작...")
    train_dataset = preprocess_for_ner(new_data_df, tokenizer)
    
    if train_dataset is None:
        logging.warning("⚠️ 전처리 후 학습할 유효한 데이터가 없습니다. 학습을 건너뜁니다.")
        # (참고: PII를 context에서 못 찾는 등의 이유로 데이터가 0이 될 수 있음)
        return
        
    logging.info(f"✅ 데이터 전처리 완료. (유효 샘플: {len(train_dataset)})")

    # 5. (✨ 신규) 실제 학습(Fine-Tuning) 시작
    
    # (✨ 패딩 오류 수정)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # (NCP 서버 사양에 맞춰 최소한의 설정으로 학습)
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_PATH, "checkpoints"), # 학습 중간 과정 저장
        num_train_epochs=3,             # 3번 반복 학습
        per_device_train_batch_size=2,  # 한 번에 2개씩 (CPU/저사양 GPU용)
        save_strategy="epoch",          # 1 에포크마다 저장
        logging_steps=10,               # 10 스텝마다 로그 출력
        report_to="none",               # (필수) wandb 같은 외부 로깅 비활성화
        
        # (✨✨✨ 핵심 수정: 용량 관리 ✨✨✨)
        # 가장 최근의 체크포인트 2개만 남기고 나머지는 자동으로 삭제합니다.
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator # (✨ 패딩 오류 수정)
    )

    logging.info("🔥 '경력직' 뇌 실제 학습 시작... (CPU/GPU 사용)")
    trainer.train()
    logging.info("✅ 재학습 완료!")

    # 6. (✨ 신규) '경력직' 뇌 최종 저장
    logging.info(f"💾 '경력직' 뇌를 {MODEL_PATH}에 저장합니다.")
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH) # (중요) 토크나이저도 함께 저장

    # 7. (기존) "학습 완료"된 ID들을 로그에 기록 (중복 학습 방지)
    for unique_id in new_ids_to_log:
        save_trained_log(unique_id)
        
    logging.info(f"💾 {len(new_ids_to_log)}건을 '학습 완료' 처리했습니다.")
    logging.info("🤖 3. '학습기' 봇(Trainer) 작동 완료.")

if __name__ == "__main__":
    main()
