import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import BertTokenizer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置参数
class Config:
    train_path = "./data/translation2019zh_train.json"
    batch_size = 32
    d_model = 512
    max_len = 50
    n_head = 8
    ffn_hidden = 2048
    n_layers = 6
    dropout = 0.1
    lr = 1e-4
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据集类
import random
class TranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, trg_tokenizer, max_len,max_samples=50000):
        self.data = []
        # 先读取全部数据
        with open(file_path, "r", encoding="utf-8") as f:
            full_data = [json.loads(line) for line in f]

        # 随机采样
        if max_samples and max_samples < len(full_data):
            self.data = random.sample(full_data, max_samples)
        else:
            self.data = full_data
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]["english"]
        trg_text = self.data[idx]["chinese"]

        src_enc = self.src_tokenizer(
            src_text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )
        trg_enc = self.trg_tokenizer(
            trg_text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )

        return {
            "src_ids": src_enc["input_ids"],
            "trg_input_ids": trg_enc["input_ids"][:-1],
            "trg_output_ids": trg_enc["input_ids"][1:]
        }


# 数据加载器
def collate_fn(batch, src_pad_id, trg_pad_id):
    src_ids = [item["src_ids"] for item in batch]
    trg_input = [item["trg_input_ids"] for item in batch]
    trg_output = [item["trg_output_ids"] for item in batch]

    def pad_sequence(sequences, pad_id):
        max_len = max(len(s) for s in sequences)
        return [s + [pad_id] * (max_len - len(s)) for s in sequences]

    return {
        "src": torch.LongTensor(pad_sequence(src_ids, src_pad_id)),
        "trg_input": torch.LongTensor(pad_sequence(trg_input, trg_pad_id)),
        "trg_output": torch.LongTensor(pad_sequence(trg_output, trg_pad_id))
    }


# 初始化分词器
src_tokenizer = BertTokenizer.from_pretrained("my_tokenizers/bert-base-uncased")
trg_tokenizer = BertTokenizer.from_pretrained("my_tokenizers/bert-base-chinese")

# 数据加载
dataset = TranslationDataset(
    Config.train_path,
    src_tokenizer,
    trg_tokenizer,
    Config.max_len,
    max_samples=200000
)
train_loader = DataLoader(
    dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    collate_fn=partial(
        collate_fn,
        src_pad_id=src_tokenizer.pad_token_id,
        trg_pad_id=trg_tokenizer.pad_token_id
    )
)

# 初始化模型
from model1 import *
model = Transformer(
    src_pad_idx=src_tokenizer.pad_token_id,
    trg_pad_idx=trg_tokenizer.pad_token_id,
    enc_voc_size=src_tokenizer.vocab_size,
    dec_voc_size=trg_tokenizer.vocab_size,
    d_model=Config.d_model,
    max_len=Config.max_len,
    n_head=Config.n_head,
    ffn_hidden=Config.ffn_hidden,
    n_layers=Config.n_layers,
    dropout=Config.dropout
).to(Config.device)

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.pad_token_id)

# 训练循环
for epoch in range(Config.epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        src = batch["src"].to(Config.device)
        trg_input = batch["trg_input"].to(Config.device)
        trg_output = batch["trg_output"].to(Config.device)

        optimizer.zero_grad()
        output = model(src, trg_input)
        loss = criterion(
            output.view(-1, output.size(-1)),
            trg_output.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch + 1}/{Config.epochs} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "transformer_model.pth")