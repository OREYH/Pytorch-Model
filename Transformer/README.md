# Korean–Japanese Transformer Machine Translation

> SentencePiece 기반 토크나이저와 커스텀 Transformer 아키텍처로 구현한 한국어–일본어 기계번역기 프로젝트이다.

## 1. 프로젝트 소개
- 한국어 문장을 자연스럽게 일본어로 번역하기 위한 end-to-end Transformer 파이프라인을 제공한다.
- SentencePiece로 공유 서브워드 사전을 학습하고, 직접 구현한 Encoder-Decoder Transformer를 사용해 한국어→일본어 번역을 지원한다.
- 학습 파이프라인은 Noam warmup 스케줄러, mixed precision, gradient clipping·accumulation을 포함해 논문 수준 설정을 재현한다.
- 모든 설정은 TOML(`config/config.toml`)에서 제어하며, `main.py` 하나로 학습과 추론 모드를 전환한다.

## 2. 주요 기능
- 한국어–일본어 채팅 JSON 데이터를 라인 단위 병렬 텍스트로 전처리한다.
- SentencePiece 토크나이저를 학습하고 필요한 곳에서 로드한다.
- 커스텀 Multi-Head Attention·Positional Encoding·Encoder/Decoder 블록으로 구성된 Transformer를 학습/평가한다.
- 학습된 체크포인트(`best.pt`)를 로드해 문장 단위 greedy 번역을 수행한다.

## 3. 디렉터리 구조
```text
.
├── main.py
├── scripts/
│   ├── build_tokenizer.py
│   ├── dataloader.py
│   ├── model.py
│   ├── notify_ntfy.py
│   └── train.py
├── config/
│   └── config.toml
├── data/
│   ├── chat.json
│   ├── text/
│   └── tokenizer/
├── checkpoints/
├── raw_data/
└── README.md
```
- `main.py`: 설정 로드, 토크나이저/데이터로더/모델 초기화, train·inference 모드 실행을 담당.
- `scripts/build_tokenizer.py`: JSON 병렬 데이터를 TXT로 변환하고 SentencePiece 학습/테스트를 자동화.
- `scripts/dataloader.py`: SentencePiece 로더, 텍스트 데이터셋, Collator, `create_loader`를 정의.
- `scripts/model.py`: MultiHeadAttention부터 Transformer 본체까지 PyTorch로 구현.
- `scripts/train.py`: Noam 스케줄러, 학습/평가 루프, auto-regressive `inference`를 포함.
- `scripts/notify_ntfy.py`: ntfy 푸시 알림을 보내는 데코레이터를 제공.
- `config/config.toml`: data/train/model/inference 섹션으로 구성된 기본 설정.
- `data/`: 원본 `chat.json`, 변환된 `text/*.txt`, SentencePiece 산출물(`tokenizer/*.model|*.vocab`)을 보관.
- `checkpoints/`: 학습 시 `best.pt`를 저장.
- `raw_data/`: 추가 정제 전 데이터를 둘 수 있는 디렉터리.

## 4. 요구 사항(Requirements)
- Python 3.11 이상을 권장.
- 주요 패키지는 torch>=2.x, sentencepiece, tomli, tqdm.
- 설치 명령은 다음과 같다.

```bash
pip install -r requirements.txt
# 또는
pip install torch sentencepiece tomli tqdm
```

## 5. 데이터 준비

### 5.1 원본 JSON 형식
- `./data/chat.json`은 `{"원문": "...", "최종번역문": "..."}` 형식으로 한국어 원문과 일본어 최종 번역문을 저장한다.
- `scripts/build_tokenizer.py`가 JSON을 로드한 뒤 길이의 97%를 train, 나머지를 valid/test로 반반 나누고 seed=42로 셔플한다.

### 5.2 텍스트 파일 및 토크나이저 학습
- TXT는 한국어 한 줄, 이어서 같은 쌍의 일본어 한 줄 순으로 `(KOR, JPN)`이 번갈아 나온다.
- 실행 명령:

```bash
python scripts/build_tokenizer.py
```

- 이 스크립트는 다음을 수행한다.
  1. `./data/text/chat_train.txt`, `chat_valid.txt`, `chat_test.txt`를 생성하고 `kor\n`→`jpn\n` 순서로 반복 저장한다.
  2. `chat_train.txt`를 사용해 SentencePiece를 학습한다. 기본 설정은 `vocab_size=32000`, `character_coverage=0.9995`, `<bos>=0`, `<eos>=1(</s>)`, `<pad>=2(<pad>)`, `<unk>=3(<unk>)`다.
  3. 학습된 모델(`./data/tokenizer/sp_kor_jpn.model`, `.vocab`)을 로드해 한-일 샘플 문장을 Encode/Decode하며 sanity check를 출력한다.

### 5.3 DataLoader 구성
- `load_tokenizer(tokenizer_path)`는 SentencePiece 모델을 로드한 `SentencePieceProcessor`를 반환한다. 기본 경로는 `./data/tokenizer/sp_kor_jpn.model`이다.
- `TextDataset`은 TXT 라인을 `(lines[0], lines[1])` 식으로 `(한국어, 일본어)` 쌍으로 묶고, SentencePiece ID로 변환한다. Encoder 입력은 `src_ids`, Decoder 입력은 `<bos> + tgt_ids`, Decoder 정답은 `tgt_ids + <eos>`로 분리한다.
- `Collator`는 `(batch, src_len)`·`(batch, tgt_len)` 텐서를 PAD 아이디로 초기화하고 실제 길이만큼 덮어쓴 뒤 PAD 위치가 True인 `src_pad_mask`, `tgt_pad_mask`를 만든다.
- `create_loader(...)`는 위 데이터셋/Collator를 조합해 `pin_memory=True`인 `DataLoader`를 생성한다. 예제대로라면 train 로더는 `shuffle=True`, valid/test 로더는 `False`로 두고 `chat_train/valid/test.txt` 경로를 넘긴다.

## 6. 설정 파일(`config/config.toml`) 설명

| 섹션 | 키 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `[data]` | `data_dir` | `"./data"` | 텍스트/토크나이저 루트
|  | `src_len` | `64` | Encoder 입력 최대 길이
|  | `tgt_len` | `64` | Decoder 입력/정답 최대 길이
| `[train]` | `batch_size` | `256` | 학습 배치 크기
|  | `val_batch_size` | `256` | 검증 배치 크기
|  | `epochs` | `20` | 학습 epoch 수
|  | `warmup_steps` | `4000` | Noam WarmUp 스텝 수
|  | `accumulation_steps` | `1` | gradient accumulation 스텝
|  | `grad_clip` | `1.0` | gradient clipping max norm
|  | `weight_decay` | `0.0001` | Adam weight decay
|  | `label_smoothing` | `0.1` | CrossEntropy label smoothing
|  | `lr_factor` | `1.0` | Noam 스케줄러 factor
|  | `seed` | `42` | 재현성을 위한 시드다
|  | `save_dir` | `"./checkpoints"` | `best.pt` 저장 경로
| `[model]` | `d_model` | `512` | 임베딩/모델 차원
|  | `nhead` | `8` | Multi-head 개수
|  | `num_encoder_layers` | `6` | Encoder 레이어 수
|  | `num_decoder_layers` | `6` | Decoder 레이어 수
|  | `dim_feedforward` | `2048` | FFN 내부 차원
|  | `dropout` | `0.1` | Dropout 비율
|  | `max_seq_len` | `256` | positional encoding 최대 길이
| `[inference]` | `sample_text` | `"먼저 퇴근할게요."` | CLI inference 기본 입력 문장
|  | `max_infer_len` | `64` | greedy decoding 최대 길이

예시 스니펫:

```toml
[data]
data_dir = "./data"
src_len = 64
tgt_len = 64

[train]
batch_size = 256
val_batch_size = 256
epochs = 20
warmup_steps = 4000
accumulation_steps = 1
grad_clip = 1.0
weight_decay = 0.0001
label_smoothing = 0.1
lr_factor = 1.0
seed = 42
save_dir = "./checkpoints"
```

## 7. 학습(Training)

### 7.1 사용법(Usage) & Quick Start
| 인자 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--config` | str | `config.toml` | TOML 경로. 루트의 `config.toml`을 기본으로 찾지만 일반적으로 `config/config.toml`을 지정.
| `--mode` | str (`train`/`inference`) | `None` | 학습 또는 추론 모드. 미지정 시 argparse가 에러를 발생.

필수 실행 명령은 다음과 같다.

```bash
# 1) 토크나이저 학습 및 텍스트 데이터 생성
python scripts/build_tokenizer.py

# 2) Transformer 모델 학습
python main.py --config config/config.toml --mode train

# 3) 학습된 모델로 번역 실행
python main.py --config config/config.toml --mode inference
```
- 1)은 JSON→TXT 변환과 SentencePiece 학습/검증을 수행한다.
- 2)는 토크나이저 로드 → `create_loader`로 train/valid 로더 구성 → Transformer/손실/Adam/Noam/GradScaler 초기화 → epoch 루프 → `checkpoints/best.pt` 저장까지 진행한다.
- 3)은 `best.pt`를 로드한 뒤 `inference.sample_text`를 greedy decoding해 터미널로 출력한다.

### 7.2 학습 파이프라인 설명
- `main.py`는 TOML에서 `[data]`, `[train]`, `[model]`, `[inference]`를 각각 경로·로더 하이퍼파라미터·모델 옵션·추론 옵션으로 분리한다.
- 토크나이저는 `data_dir/tokenizer/sp_kor_jpn.model` 경로에서 로드하고, `create_loader`로 train/valid DataLoader를 만든다.
- 손실은 PAD 토큰을 무시하는 `CrossEntropyLoss`와 label smoothing, 옵티마이저는 `Adam(beta=(0.9,0.98), eps=1e-9)`다.
- `train()`은 `torch.amp.autocast`로 mixed precision을 수행하고 `amp.GradScaler`가 loss를 스케일링한다. `accumulation_steps`마다만 optimizer step을 하고, 그 전에 `scaler.unscale_`로 복구해 `nn.utils.clip_grad_norm_(max_norm)`으로 gradient를 자른다. PAD가 아닌 토큰 수(`tgt_out.ne(pad_id)`)로 손실을 정규화해 epoch 평균 loss와 `math.exp(loss)` 기반 perplexity를 계산한다.
- `evaluate()`는 `torch.no_grad()`와 `autocast`로 검증을 돌리고 손실/Perplexity를 동일 방식으로 계산한다.
- `NoamScheduler`는 `factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)` 공식을 사용해 LR을 조정한다.
- 학습 루프는 `[Epoch xx] train_loss=... train_ppl=... valid_loss=... valid_ppl=... lr=...` 형식으로 로그를 남기며, 검증 손실이 개선되면 `model_state`, `optimizer_state`, `scheduler_state(step, lr)`, `scaler_state`, `config`를 묶어 `best.pt`로 저장한다.
- 로그 예시: `[Epoch 01] train_loss=3.8123 train_ppl=45.24 valid_loss=3.6051 valid_ppl=36.81 lr=0.000975`

### 7.3 Transformer 기본 하이퍼파라미터
| 항목 | 값/설명 |
| --- | --- |
| `vocab_size` | SentencePiece `sp_kor_jpn.model` 크기(기본 32k지만 학습 결과에 따라 달라짐)
| `pad_id` | SentencePiece `<pad>` id(기본 2)이며 embedding padding_idx에 사용
| `d_model` | 512 (config로 변경 가능)
| `nhead` | 8
| `num_encoder_layers` / `num_decoder_layers` | 6 / 6
| `dim_feedforward` | 2048이며 각 레이어 두 번째 Linear 입력 크기
| `dropout` | 0.1로 embedding/attention/FFN에 적용
| `max_seq_len` | 256으로 PositionalEncoding buffer 길이
| 입력/출력 shape | `src_ids` `[B, src_len]`, `tgt_ids` `[B, tgt_len]`, 출력 `logits` `[B, tgt_len, vocab_size]`
| 마스킹 | Decoder는 상삼각 `-inf` causal mask와 PAD mask를 함께 적용해 미래 토큰과 PAD를 무시

- `MultiHeadAttention`은 Q/K/V 투영, head 분할·병합, look-ahead 및 key padding mask 적용을 직접 처리
- `EncoderLayer`는 self-attn + FFN을 pre/post norm 옵션에 맞춰 실행
- `DecoderLayer`는 self-attn → cross-attn → FFN 순으로 encoder memory를 활용
- `Encoder`·`Decoder`는 위 레이어를 `ModuleList`로 쌓은 뒤 마지막에 LayerNorm을 적용
- `PositionalEncoding`은 sinusoidal 패턴을 buffer로 등록해 임베딩에 더함

## 8. 추론(Inference)
- `--mode inference`로 실행하면 `checkpoints/best.pt`를 로드하고 TOML의 `inference.sample_text`, `max_infer_len`을 사용한다.
- `scripts/train.py`의 `inference` 함수는 SentencePiece로 입력 문장을 ID로 변환한 뒤 `<bos>`에서 시작해 greedy decoding을 수행하고 `<eos>`를 만나면 종료한다.
- CLI 예시는 아래와 같다.

```bash
python main.py --config config/config.toml --mode inference
입력: 먼저 퇴근할게요.
번역: 先に退勤します。
```

- 코드에서 직접 호출할 수도 있다.

```python
from scripts.dataloader import load_tokenizer
from scripts.model import Transformer
from scripts.train import inference
import torch, os, tomli

cfg = tomli.load(open("config/config.toml", "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = load_tokenizer(os.path.join(cfg["data"]["data_dir"], "tokenizer", "sp_kor_jpn.model"))
model = Transformer(vocab_size=tokenizer.GetPieceSize(), pad_id=tokenizer.pad_id(), **cfg["model"]).to(device)
ckpt = torch.load(os.path.join(cfg["train"]["save_dir"], "best.pt"), map_location=device)
model.load_state_dict(ckpt["model_state"])
print(inference(model, tokenizer, "번역하고 싶은 문장", device, max_len=64))
```

## 9. 스크립트별 역할 정리
| 스크립트 | 역할 |
| --- | --- |
| `scripts/build_tokenizer.py` | JSON→TXT 변환, SentencePiece 학습/테스트를 처리
| `scripts/dataloader.py` | SentencePiece 로드, `(kor, jpn)` 라인 쌍 토큰화, PAD/Look-ahead 마스크 생성을 담당
| `scripts/model.py` | MultiHeadAttention, PositionalEncoding, Encoder/Decoder, Transformer 본체를 정의
| `scripts/train.py` | Noam 스케줄러, mixed precision 학습, gradient accumulation/clipping, 평가, auto-regressive inference를 구현
| `scripts/notify_ntfy.py` | `@ntfy_notify` 데코레이터로 학습/추론 알림을 보냄
| `main.py` | argparse 인자 처리, 설정 로드, 토크나이저/로더/모델 초기화, train·inference 플로우 전환을 담당

## 10. 전체 파이프라인 요약
1. `python scripts/build_tokenizer.py` → JSON 병렬 데이터를 라인 단위 TXT로 변환하고 SentencePiece(`sp_kor_jpn.model/.vocab`)를 학습
2. `python main.py --config config/config.toml --mode train` → 토크나이저 로드 → `create_loader` 구성 → Transformer + 손실/Adam/Noam/GradScaler 설정 → mixed precision 학습 → `checkpoints/best.pt` 저장으로 이어진다.
3. `python main.py --config config/config.toml --mode inference` → `best.pt` 로드 → SentencePiece로 `sample_text` 토큰화 → greedy decoding으로 일본어 문장을 출력한다.
