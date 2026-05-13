import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import re
import os

pad_token = "<|endoftext|>"


def build():
    model_name_or_path = "Qwen/Qwen-1_8B"

    # 1. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load from HF: {e}")
        exit(1)

    # 2. 设置特殊 token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"Qwen Tokenizer loaded. Vocab Size: {vocab_size}")
    print(
        f"Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}"
    )

    dataset_path = "./dataset/"
    file_names = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]

    if not file_names:
        print("No .txt files found in dataset directory.")
        return

    for file_name in file_names:
        input_file_path = os.path.join(dataset_path, file_name)
        # 生成对应的 pt 文件名 (例如: data.txt -> data.pt)
        output_file_name = file_name.replace(".txt", ".pt")
        output_file_path = os.path.join(dataset_path, output_file_name)

        print(f"Processing: {file_name}...")

        # 读取所有行
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 去除每行末尾的换行符 (可选，tokenizer通常能处理，但去除更干净)
        lines = [line.rstrip("\n") for line in lines]

        # 3. 编码所有行
        # tokens_list 是一个列表，包含所有行的编码结果 (list of list of ints)
        tokens_list = [
            tokenizer.encode(line, add_special_tokens=False) for line in lines
        ]

        # 4. 计算整个文件中的最大长度，用于统一补齐
        # 这样可以保证最终生成的 Tensor 是规则的 (Rectangular)
        max_len = max(len(t) for t in tokens_list)

        # 5. 补齐所有序列到 max_len
        padded_tokens = []
        for t in tokens_list:
            padding_length = max_len - len(t)
            # 在右侧补齐 pad_token_id
            padded_t = t + [tokenizer.pad_token_id] * padding_length
            padded_tokens.append(padded_t)

        # 7. 转换为 Torch Tensor
        # 形状: (num_pairs, 1, max_len)
        tensor_data = torch.tensor(padded_tokens, dtype=torch.long)

        # 8. 保存
        torch.save(tensor_data, output_file_path)
        print(f"  Saved: {output_file_path} | Shape: {tensor_data.shape}")


def fix():
    dataset_path = "./dataset/"
    file_names = os.listdir(dataset_path)
    for file_name in file_names:
        if ".jsonl" in file_name:
            with open(dataset_path + file_name, "r", encoding="utf-8") as f:
                text = f.read()
                f.close()
                contents = re.findall(r'"content": "(.+?)"', text)
                name = os.path.splitext(os.path.basename(file_name))[0]
                with open(dataset_path + name + ".txt", "w", encoding="utf-8") as fp:
                    i = 0
                    for content in contents:
                        if (i + 1) % 2 == 0:
                            fp.write(content + pad_token + "\n")
                        else:
                            fp.write(content)
                        i += 1
                    fp.close()


def fix_pt():
    dataset_path = "./dataset/"
    file_names = os.listdir(dataset_path)
    flatten = nn.Flatten(-2, -1)
    for file_name in file_names:
        if ".pt" in file_name:
            data = torch.load(dataset_path + file_name)
            print(data.shape)


def pad_to_max(tensor, target_len, pad_value=0):
    current_len = tensor.shape[-1]
    if current_len == target_len:
        return tensor

    # 计算需要填充的数量 (只在最后一维后面补)
    pad_amount = target_len - current_len

    # F.pad 的 padding 参数是从后往前定义的: (dim_1_end, dim_1_start, dim_0_end, dim_0_start...)
    # 我们只需要在最后一种维度 (dim -1) 的末尾补充，所以是 (0, pad_amount)
    # 前面的维度不需要补，所以填 0
    padding = (0, pad_amount)

    return F.pad(tensor, padding, mode="constant", value=pad_value)


def pad():
    model_name_or_path = "Qwen/Qwen-1_8B"

    # 1. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load from HF: {e}")
        exit(1)

    # 2. 设置特殊 token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"Qwen Tokenizer loaded. Vocab Size: {vocab_size}")
    print(
        f"Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}"
    )

    dataset_path = "./dataset/"
    file_names = os.listdir(dataset_path)
    for file_name in file_names:
        if ".pt" in file_name:
            data = torch.load(dataset_path + file_name)
            data = pad_to_max(data, 301, tokenizer.pad_token_id)
            print(data.shape)
            torch.save(data, dataset_path + file_name)


def full_build(L=128001, dataset_path="./dataset/"):
    """
    读取 dataset/ 下的所有 .txt 文件，将每个文件编码为 token ids，
    并分割成固定长度 L 的片段（不足 L 的丢弃），保存为 .pt 张量。

    Args:
        L: 每个样本的序列长度（包含最后一个 token 用于错位生成标签）
    """
    model_name_or_path = "../tokenizer/"

    # 1. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load from HF: {e}")
        exit(1)
    vocab_size = len(tokenizer)
    print(f"Qwen Tokenizer loaded. Vocab Size: {vocab_size}")
    print(
        f"Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}"
    )

    file_names = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]

    if not file_names:
        print("No .txt files found in dataset directory.")
        return

    for file_name in file_names:
        input_file_path = os.path.join(dataset_path, file_name)
        output_file_name = file_name.replace(".txt", ".pt")
        output_file_path = os.path.join(dataset_path, output_file_name)

        print(f"Processing: {file_name}...")

        # 读取整个文件内容
        text = ""
        with open(input_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.replace("\n", "")
                text += "<|im_start|>" + line + "<|im_end|>"

        # 编码为 token ids
        tokens = tokenizer.encode(text, add_special_tokens=False)

        total_tokens = len(tokens)
        print(f"  Total tokens: {total_tokens}")

        # 分割为固定长度 L 的片段（不足 L 的丢弃）
        num_chunks = total_tokens // L
        if num_chunks == 0:
            print(f"  Warning: file too short (< {L}), skipping.")
            continue

        chunks = []
        for i in range(num_chunks):
            start = i * L
            end = start + L
            chunk = tokens[start:end]
            chunks.append(chunk)

        # 转换为 PyTorch 张量
        tensor_data = torch.tensor(chunks, dtype=torch.long)  # shape: (num_chunks, L)

        # 保存
        torch.save(tensor_data, output_file_path)
        print(f"  Saved: {output_file_path} | Shape: {tensor_data.shape}")


if __name__ == "__main__":
    full_build(129, dataset_path="./")
