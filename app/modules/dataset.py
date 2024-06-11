import torch
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, sentences, labels_taskA, labels_taskB, tokenizer, max_len):
        self.sentences = sentences
        self.labels_taskA = labels_taskA
        self.labels_taskB = labels_taskB
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels_taskA = self.labels_taskA[idx]
        labels_taskB = self.labels_taskB[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        dataset={
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_taskA': torch.tensor(labels_taskA, dtype=torch.long),
            'labels_taskB': torch.tensor(labels_taskB, dtype=torch.float),
        }
        return dataset
