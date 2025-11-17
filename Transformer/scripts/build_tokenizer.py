"""
1. train/valid/test TXT 파일 생성
2. train data를 이용해 SentencePiece 모델 학습
3. SentencePiece 모델 테스트
"""

#### 라이브러리 호출 ####
import os, json
import random
import sentencepiece as spm

def convert_data_to_txt(data_path, train_ratio=0.97, seed=42):
    """ 학습용 TXT 파일 생성 """
    # './data/text'에 저장
    
    # 저장 경로 생성
    save_dir = os.path.join(os.path.dirname(data_path), 'text') # './data/text'
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON 파일 불러오기
    data = json.load(open(data_path, "r", encoding="utf-8"))
    
    # 데이터 섞기
    random.seed(seed)
    random.shuffle(data)
    
    # data별 길이 계산
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_valid = (n_total - n_train) // 2
    
    # 데이터 분할
    train_data = data[:n_train]
    valid_data = data[n_train:n_train + n_valid]
    test_data  = data[n_train + n_valid:]

    mapping_data = { 'train': train_data, 'valid': valid_data, 'test': test_data }
    
    # TXT 파일로 변환
    for key, data in mapping_data.items():
        txt_path = os.path.join(save_dir, f'chat_{key}.txt')
        
        # 파일 초기화
        with open(txt_path, 'w', encoding='utf-8') as file:
            pass
        # 한국어-일본어 번갈아 저장
        with open(txt_path, 'a', encoding='utf-8') as file:
            for item in data:
                kor_str = item['원문'].strip()
                jap_str = item['최종번역문'].strip()
                file.write(kor_str + '\n')
                file.write(jap_str + '\n')

def train_sentencepiece(data_path, output_dir='./data/tokenizer',
                        model_prefix='sp_kor_jpn',
                        bos_id=0, eos_id=1, pad_id=2, unk_id=3, 
                        vocab_size=32000, character_coverage=0.9995):
    """ SentencePiece 모델 학습 """
    
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, model_prefix)
    
    spm.SentencePieceTrainer.Train(
        f'--input={data_path} '
        f'--model_prefix={prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={character_coverage} '
        f'--bos_id={bos_id} '
        f'--eos_id={eos_id} '
        f'--pad_id={pad_id} '
        f'--unk_id={unk_id} '
        f'--eos_piece=</s> '
        f'--pad_piece=<pad> '
        f'--unk_piece=<unk>')

def test_sentencepiece(model_path, sentences):
    """ SentencePiece 모델 테스트 """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    for sentence in sentences:
        # Encoding: 문장을 학습된 Ids로 변환
        ids = sp.EncodeAsIds(sentence)
        # Decoding: Ids를 문장으로 변환
        decoded = sp.DecodeIds(ids)
        # Token: 문장을 분할된 token으로 변환
        tokens = sp.EncodeAsPieces(sentence)
        
        print("Original:", sentence)
        print("Tokens:", tokens)
        print("IDs:", ids)
        print("Decoded:", decoded)
        print("----")
    
if __name__ == '__main__':
    data_dir  = './data'
    # 1. TXT 파일 생성
    json_path = os.path.join(data_dir, 'chat.json')
    convert_data_to_txt(json_path)
    
    # 2. SentencePiece 모델 학습
    data_path = os.path.join(data_dir, 'text', 'chat_train.txt')
    train_sentencepiece(data_path) 
    
    # 3. SentencePiece 모델 테스트
    sentences = [
        "저는 한국인입니다.",
        "僕は韓国人です。" ]
    
    model_path = os.path.join(data_dir, 'tokenizer', 'sp_kor_jpn.model')
    test_sentencepiece(model_path, sentences)