# 이 파일은 DataLoader에 들어갈 Dataset Class를 정의합니다.
# Dataset에 필요한 function으로는 __getitem__, __len__ 이 있습니다.

import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, data_root="../Data/archive", sample_rate=16000, n_mfcc=40):
        self.samples = []
        data_root = Path(data_root).resolve()
        with open(list_file) as f:
            for line in f:
                path, label = line.strip().split(",")
                self.samples.append((f"{data_root}/{path}", label))

        self.classes = ["yes", "no", "stop", "go", "silence", "other"]
        self.label_map = {"yes": 0, "no": 1, "stop": 2, "go": 3, "silence": 4, "other": 5}
        self.sample_rate = sample_rate

        # 오디오 전처리: MFCC 변환기
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        
        # 1. 오디오 불러오기 (waveform shape: [1, N])
        waveform, sr = torchaudio.load(path)
        
        # 2. 샘플레이트 맞추기 (간혹 다를 수 있음)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 3. 길이 맞추기 (1초 = 16000)
        if waveform.size(1) < self.sample_rate:
            pad = self.sample_rate - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.sample_rate]

        # 4. MFCC 변환
        mfcc = self.mfcc_transform(waveform)  # shape: [1, n_mfcc, time]
        return mfcc, self.label_map[label_str]

    def __len__(self):
        return len(self.samples)


"""
사용예시
train_dataset = SpeechDataset("train_list.txt")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

for x, y in train_loader:
    print(x.shape)  # [B, 1, n_mfcc, time]
    print(y)        # [B] 클래스 index
    break

"""
