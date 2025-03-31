from transformers import BertTokenizer
import torch
from model1 import *

def translate(model, src_sentence, src_tokenizer, trg_tokenizer, max_len=50, device="cpu"):
    model.eval()
    # 编码源文本
    src_enc = src_tokenizer(
        src_sentence,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(device)

    # 初始化目标输入
    trg_input = torch.LongTensor([[trg_tokenizer.cls_token_id]]).to(device)

    for _ in range(max_len):
        # 生成预测
        with torch.no_grad():
            output = model(src_enc.input_ids, trg_input)

        # 获取最后一个预测的token
        next_token = output.argmax(dim=-1)[:, -1]
        trg_input = torch.cat([trg_input, next_token.unsqueeze(0)], dim=1)

        # 遇到[SEP]停止生成
        if next_token.item() == trg_tokenizer.sep_token_id:
            break

    # 解码目标序列
    trg_ids = trg_input.squeeze().tolist()
    translation = trg_tokenizer.decode(trg_ids, skip_special_tokens=True)
    return translation


# 加载模型
config = {
    "d_model": 512,
    "max_len": 50,
    "n_head": 8,
    "ffn_hidden": 2048,
    "n_layers": 6,
    "dropout": 0.1
}

import os
src_tokenizer = BertTokenizer.from_pretrained("my_tokenizers/bert-base-uncased")
trg_tokenizer = BertTokenizer.from_pretrained("my_tokenizers/bert-base-chinese")

model = Transformer(
    src_pad_idx=src_tokenizer.pad_token_id,
    trg_pad_idx=trg_tokenizer.pad_token_id,
    enc_voc_size=src_tokenizer.vocab_size,
    dec_voc_size=trg_tokenizer.vocab_size,
    **config
).to("cpu")
model.load_state_dict(torch.load("transformer_model.pth", map_location="cpu"))
print(model.state_dict().keys())
# 示例翻译
src_sentence = "i love china,it is the most beautiful country i have gone"
translation = translate(
    model,
    src_sentence,
    src_tokenizer,
    trg_tokenizer,
    device="cpu"
)
print(f"Source: {src_sentence}")
print(f"Translation: {translation}")