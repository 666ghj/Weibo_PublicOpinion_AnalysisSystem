import os
import sys
import json
import re
import argparse
from typing import Dict, Tuple, List

# ========== 单卡锁定（在导入 torch/transformers 前执行） ==========
def _extract_gpu_arg(argv, default: str = "0") -> str:
    for i, arg in enumerate(argv):
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
        if arg == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
    return default

env_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
try:
    gpu_to_use = _extract_gpu_arg(sys.argv, default="0")
except Exception:
    gpu_to_use = "0"
if (not env_vis) or ("," in env_vis):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
    os.environ.pop(_k, None)

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)


def preprocess_text(text: str) -> str:
    return text


def ensure_base_model_local(model_name_or_path: str, local_model_root: str) -> Tuple[str, AutoTokenizer]:
    os.makedirs(local_model_root, exist_ok=True)
    base_dir = os.path.join(local_model_root, "bert-base-chinese")

    def is_ready(path: str) -> bool:
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

    if is_ready(base_dir):
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        return base_dir, tokenizer

    # 本机缓存
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        base = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
        os.makedirs(base_dir, exist_ok=True)
        tokenizer.save_pretrained(base_dir)
        base.save_pretrained(base_dir)
        return base_dir, tokenizer
    except Exception:
        pass

    # 远程下载
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base = AutoModel.from_pretrained(model_name_or_path)
    os.makedirs(base_dir, exist_ok=True)
    tokenizer.save_pretrained(base_dir)
    base.save_pretrained(base_dir)
    return base_dir, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地/缓存/远程加载的中文 BERT 分类模型进行预测")
    parser.add_argument("--model_root", type=str, default="./model", help="本地模型根目录")
    parser.add_argument("--finetuned_subdir", type=str, default="bert-chinese-classifier", help="微调结果子目录")
    parser.add_argument("--pretrained_name", type=str, default="google-bert/bert-base-chinese", help="预训练模型名称或路径")
    parser.add_argument("--text", type=str, default=None, help="直接输入一条要预测的文本")
    parser.add_argument("--interactive", action="store_true", help="进入交互式预测模式")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="指定单卡 GPU，如 0 或 1")
    return parser.parse_args()


def load_finetuned(model_root: str, subdir: str) -> Tuple[str, Dict[int, str]]:
    finetuned_path = os.path.join(model_root, subdir)
    if not os.path.isdir(finetuned_path):
        raise FileNotFoundError(
            f"未找到微调模型目录: {finetuned_path}，请先运行训练脚本。"
        )
    label_map_path = os.path.join(finetuned_path, "label_map.json")
    id2label = None
    if os.path.isfile(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            id2label = {int(k): str(v) for k, v in data.get("id2label", {}).items()}
    return finetuned_path, id2label


def predict_topk(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device, text: str, max_length: int = 128, top_k: int = 3) -> List[Tuple[str, float]]:
    processed = preprocess_text(text or "")
    encoded = tokenizer(
        processed,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        k = min(top_k, probs.shape[-1])
        confs, idxs = torch.topk(probs, k)
    id2label = getattr(model.config, "id2label", {}) if isinstance(getattr(model.config, "id2label", None), dict) else {}
    results: List[Tuple[str, float]] = []
    for i in range(k):
        idx = int(idxs[i].item())
        conf = float(confs[i].item())
        label_name = id2label.get(idx, str(idx))
        results.append((label_name, conf))
    return results


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = args.model_root if os.path.isabs(args.model_root) else os.path.join(script_dir, args.model_root)
    os.makedirs(model_root, exist_ok=True)

    # 确保基础模型在本地
    ensure_base_model_local(args.pretrained_name, model_root)

    finetuned_dir, _ = load_finetuned(model_root, args.finetuned_subdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_dir)
    model.to(device)
    model.eval()

    if args.text is not None:
        topk = predict_topk(model, tokenizer, device, args.text, args.max_length, top_k=3)
        print("Top-3 预测:")
        for rank, (label, conf) in enumerate(topk, 1):
            print(f"{rank}. {label} (p={conf:.4f})")
        return

    # 默认进入交互模式（未显式指定 --text 且未显式关闭交互）
    if args.interactive or (args.text is None):
        print("进入交互模式。输入 'q' 退出。")
        while True:
            try:
                text = input("请输入文本: ").strip()
            except EOFError:
                break
            if text.lower() == "q":
                break
            if not text:
                continue
            topk = predict_topk(model, tokenizer, device, text, args.max_length, top_k=3)
            print("Top-3 预测:")
            for rank, (label, conf) in enumerate(topk, 1):
                print(f"{rank}. {label} (p={conf:.4f})")
        return
    # 理论上不会到达这里
    print("未提供输入。")


if __name__ == "__main__":
    main()


