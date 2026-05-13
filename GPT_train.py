# -*- coding: utf-8 -*-
import gc
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from GPT import GPT, loss_tools, model_name
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":
    model_name_or_path = "./tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    vocab_size = len(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT(vocab_size, d_model=256, n_heads=4, nExperts=4, num_layers=4).to(device)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_size}")
    print(f"Model size: {model_size * 4 / 1024 / 1024:.2f} MB")  # float32 = 4 bytes

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-9,
        weight_decay=1e-4,
    )

    last_step = 0
    losses = []
    val_losses = []

    # ckpt = torch.load("checkpoints/gpt.pt")
    # losses = ckpt["losses"]
    # val_losses = ckpt["val"]
    # model.load_state_dict(ckpt["model"])
    # last_step = len(losses)

    dataset = torch.load("./xhs/train.pt", mmap=True).reshape(-1, 20, 129)
    val = torch.load("./xhs/val.pt", mmap=True).unsqueeze(1)
    total_steps = dataset.shape[0]

    avg_start = time.time()
    min_loss = 1e9
    epochs = 10
    try:
        for epoch in range(epochs):
            with tqdm(
                total=total_steps,
                initial=(last_step if epoch == 0 else 0),
                desc=f"Epoch {epoch+1}/{epochs}",
            ) as pbar:
                for step, batch in enumerate(dataset):
                    if epoch == 0 and step < last_step:
                        continue

                    batch = batch.to(device)
                    start = time.time()

                    optimizer.zero_grad()
                    loss, aux_loss = loss_tools.cross_loss(model, batch)
                    total_loss = loss + aux_loss
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    losses.append(loss.item())
                    if losses[-1] < min_loss:
                        min_loss = losses[-1]

                    val_data = val[random.randint(0, val.shape[0] - 1)].to(device)
                    val_loss, _ = loss_tools.cross_loss(model, val_data)
                    val_losses.append(val_loss.item())

                    pbar.set_postfix(
                        loss=f"{losses[-1]:.4f}",
                        aux=f"{aux_loss.item():.4f}",
                        val_loss=f"{val_loss.item():.4f}",
                        time=f"{time.time() - start:.4f}s",
                    )
                    pbar.update(1)

                    if step % 1000 == 0:
                        torch.cuda.empty_cache()
                        alloc = torch.cuda.memory_allocated() / 1024**3
                        reserv = torch.cuda.memory_reserved() / 1024**3
                        pbar.write(
                            f"[Memory] allocated: {alloc:.2f} GB, reserved: {reserv:.2f} GB"
                        )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    total_time = time.time() - avg_start
    print(f"Total time: {total_time:.4f}s | min loss: {min_loss:.4f}")

    torch.save(
        {"model": model.state_dict(), "losses": losses, "val": val_losses},
        f"checkpoints/{model_name}.pt",
    )
    print(f"saved checkpoint as /checkpoints/{model_name}.pt")

    plt.title(f"Model {model_name} Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.savefig(f"plots/{model_name}_loss.png")
    plt.show()
