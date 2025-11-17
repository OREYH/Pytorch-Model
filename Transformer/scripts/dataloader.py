""" 데이터로더 생성 """

#### 라이브러리 호출 ####
import os, glob
from types import SimpleNamespace
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader

def load_tokenizer(tokenizer_path):
    """ SentencePiece Tokeinizer 모델 불러오기 """
    sp = spm.SentencePieceProcessor()
    if sp.Load(tokenizer_path):
        return sp
    else:
        raise ModuleNotFoundError(f"{tokenizer_path}가 존재하지 않습니다!")

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer:spm.SentencePieceProcessor):
        """ 원본 데이터를 읽어서 Dataset으로 변환 """
        self.sp     = tokenizer # tokenizer model
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        
        self.data   = self.load_data(data_path) # kor-jpn 데이터 쌍으로 이루어진 리스트
        self.src_data, self.tgt_in_data, self.tgt_out_data = self.make_tokenized_data()
        
    def load_data(self, data_path):
        """ TXT 파일 불러와서 KOR-JPN 데이터 쌍 형태의 리스트로 반환 """
        with open(data_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines() # 라인 별로 불러오기
        
        # KOR-JPN 데이터쌍으로 변환
        data = [(lines[idx], lines[idx+1]) for idx in range(0, len(lines), 2)]
        
        return data

    def make_tokenized_data(self):
        """ 문장 형식의 데이터를 토큰화된 데이터로 반환 """
        src_data = []
        tgt_in_data = []
        tgt_out_data = []
        
        for src, tgt in self.data:
            src_ids = self.sp.EncodeAsIds(src)
            tgt_ids = self.sp.EncodeAsIds(tgt)
            
            # Decoder용 입력/출력 분리
            tgt_in  = [self.bos_id] + tgt_ids # <bos> 붙이기
            tgt_out = tgt_ids + [self.eos_id] # <eos> 붙이기
            
            # 리스트에 데이터 추가
            src_data.append(src_ids)     # Encoder Inputs
            tgt_in_data.append(tgt_in)   # Decoder Inputs
            tgt_out_data.append(tgt_out) # Decoder Outputs
            
        assert len(src_data) == len(tgt_in_data) == len(tgt_out_data), "데이터셋 길이가 다릅니다!"
        
        return src_data, tgt_in_data, tgt_out_data
    
    def __len__(self):
        return len(self.src_data)
        
    def __getitem__(self, idx):
        src, tgt_in, tgt_out = self.src_data[idx], self.tgt_in_data[idx], self.tgt_out_data[idx]
        
        src     = torch.tensor(src, dtype=torch.long)
        tgt_in  = torch.tensor(tgt_in, dtype=torch.long)
        tgt_out = torch.tensor(tgt_out, dtype=torch.long)
        
        return SimpleNamespace(src=src, tgt_in=tgt_in, tgt_out=tgt_out)


class Collator():
    def __init__(self, pad_id, src_len, tgt_len):
        """ Collate class로, Padding 진행 """
        self.pad_id  = pad_id
        self.src_len = src_len
        self.tgt_len = tgt_len
        
    def __call__(self, batch):
        # batch 크기에 맞는 tensor 생성
        batch_size = len(batch)
        src_ids     = torch.full((batch_size, self.src_len), self.pad_id, dtype=torch.long)
        tgt_in_ids  = torch.full((batch_size, self.tgt_len), self.pad_id, dtype=torch.long)
        tgt_out_ids = torch.full((batch_size, self.tgt_len), self.pad_id, dtype=torch.long)
        
        for idx, item in enumerate(batch):
            # src.shape: [len(srcs)] / tgt_in.shape: [len(tgt_in)] / tgt_out.shape: [len(tgt_out)]
            src, tgt_in, tgt_out = item.src, item.tgt_in, item.tgt_out
            
            src_ids[idx, :len(src)]         = src
            tgt_in_ids[idx, :len(tgt_in)]   = tgt_in
            tgt_out_ids[idx, :len(tgt_out)] = tgt_out
        
        
        # Mask Tensor 생성 (PAD 위치: True)
        src_pad_mask = (src_ids == self.pad_id)    # [B, src_len]
        tgt_pad_mask = (tgt_in_ids == self.pad_id) # [B, tgt_len]
        
        return SimpleNamespace(src_ids=src_ids, tgt_in_ids=tgt_in_ids, tgt_out_ids=tgt_out_ids, 
            src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask, 
            src_len=self.src_len, tgt_len=self.tgt_len)
        

def create_loader(data_path, tokenizer, src_len=64, tgt_len=64, batch_size=128, shuffle=True):
    """ 데이터로더 생성 """
    # 1) 데이터셋 생성
    dataset    = TextDataset(data_path, tokenizer)
    # 2) collate_fn 생성
    collate_fn = Collator(dataset.pad_id, src_len, tgt_len) 
    # 3) 데이터로더 생성
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, pin_memory=True)
    
    return loader

if __name__ == '__main__':
    # 0. 파라미터 지정
    SRC_LEN    = 64
    TGT_LEN    = 64
    BATCH_SIZE = 128
    
    # 1. 데이터 경로 지정
    data_dir = './data'
    tokenizer_path = os.path.join(data_dir, 'tokenizer', 'sp_kor_jpn.model')
    
    # 2. TXT 파일 경로 불러오기
    text_paths = glob.glob(os.path.join(data_dir, 'text', '*.txt'))
    text_paths_dict = {f"{os.path.basename(path).split('.')[0].split('_')[-1]}": path for path in text_paths}
    
    # 3. DataLoader 생성
    tokenizer = load_tokenizer(tokenizer_path)
    train_loader = create_loader(text_paths_dict['train'], tokenizer, 
                                 src_len=SRC_LEN, tgt_len=TGT_LEN, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = create_loader(text_paths_dict['valid'], tokenizer, 
                                 src_len=SRC_LEN, tgt_len=TGT_LEN, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_loader(text_paths_dict['test'], tokenizer, 
                                 src_len=SRC_LEN, tgt_len=TGT_LEN, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. DataLoader 테스트
    item = next(iter(train_loader))
    print(item.src_ids.shape)