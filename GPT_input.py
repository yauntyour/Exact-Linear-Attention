from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from GPT import GPT, model_name
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    eos_token_id=None,
    repetition_penalty=1.0,
):
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_logits = logits[:, -1] / temperature  # 1. 温度缩放

        # ---------- 重复惩罚 ----------
        if repetition_penalty != 1.0:
            prev_ids = input_ids[0]
            unique, counts = torch.unique(prev_ids, return_counts=True)
            penalty = repetition_penalty ** counts.float()
            next_logits[0, unique] /= penalty
        # ----------------------------

        # ---------- top-k 过滤 ----------
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k, dim=-1)
            next_logits[next_logits < v[:, -1:]] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    return input_ids


if __name__ == "__main__":
    model_name_or_path = "./tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    vocab_size = len(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT(vocab_size, d_model=256, n_heads=4, nExperts=4, num_layers=4).to(device)
    model.eval()

    ckpt = torch.load(f"checkpoints/{model_name}.pt")
    model.load_state_dict(ckpt["model"])

    max_new_tokens = 100
    while True:
        text = input("请输入：")
        tokens = tokenizer.encode(tokenizer.bos_token + text)
        input_len = len(tokens)
        print(f"输入长度：{input_len}")
        tokens = torch.tensor(tokens).view(1, -1).to(device)
        out = generate(
            model,
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # 降温，让分布更集中
            top_k=40,  # 只从概率最高的40个里选
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 越大惩罚越重，1.2~1.5 常用
        )

        out = out.flatten().tolist()
        print(f"输出长度：{len(out)}/{input_len+max_new_tokens}")
        print(tokenizer.decode(out))
