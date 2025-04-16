#This code is part of the ABC Music Generation project.
#This will prepare the data to pass to the transformer model.
#It will take the ABC data and convert it to a format that can be used by the transformer model.

import torch
from torch.utils.data import Dataset
import youtokentome as yttm
from parameter import (BLOCK_SIZE, DATA_FILE, TOKENIZER_MODEL600, 
                       TOKENIZER_MODEL3000, TOKENIZER_MODEL1200, 
                       DATASET600, DATASET1200, DATASET3000)

# === Dataset Class ===
class ABCDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            abc = f.read().replace("\n", " ").strip()

        # split bars
        bars = abc.split("|")
        bars = [b.strip() for b in bars if b.strip() != '']

        # group 8 input bars + 8 target bars
        for i in range(len(bars) - 16):
            input_abc = " | ".join(bars[i:i+8]) + " |"
            target_abc = " | ".join(bars[i+8:i+16]) + " |"

            input_ids = tokenizer.encode(input_abc, output_type=yttm.OutputType.ID)
            target_ids = tokenizer.encode(target_abc, output_type=yttm.OutputType.ID)

            if 16 < len(input_ids) < BLOCK_SIZE and 16 < len(target_ids) < BLOCK_SIZE:
                self.data.append({
                    'input_ids': torch.tensor(input_ids),
                    'labels': torch.tensor(target_ids)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': torch.ones_like(item['input_ids']),
            'labels': item['labels']
        }


    def save(self, output_path):
        torch.save(self.data, output_path)
        print(f" Dataset saved to {output_path}")

if __name__ == "__main__":
    tokenizer = yttm.BPE(model=TOKENIZER_MODEL600)
    dataset = ABCDataset(tokenizer, DATA_FILE)
    print(f"Dataset size: {len(dataset)}")
    dataset.save(DATASET600)
    tokenizer = yttm.BPE(model=TOKENIZER_MODEL1200)
    dataset = ABCDataset(tokenizer, DATA_FILE)
    print(f"Dataset size: {len(dataset)}")
    dataset.save(DATASET1200)
    tokenizer = yttm.BPE(model=TOKENIZER_MODEL3000)
    dataset = ABCDataset(tokenizer, DATA_FILE)
    print(f"Dataset size: {len(dataset)}")
    dataset.save(DATASET3000)