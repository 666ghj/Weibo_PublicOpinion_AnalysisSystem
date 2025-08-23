# -*- coding: utf-8 -*-
"""
Qwen3-LoRA通用训练脚本
支持0.6B、4B、8B三种规模的模型
"""
import argparse
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class Qwen3LoRAUniversal(BaseQwenModel):
    """通用Qwen3-LoRA模型"""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        super().__init__(f"Qwen3-{model_size}-LoRA")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["base_model"]
        
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_base_model(self):
        """加载Qwen3基础模型"""
        print(f"加载{self.model_size}基础模型: {self.model_name_hf}")
        
        # 第一步：检查当前文件夹的models目录
        local_model_dir = f"./models/qwen3-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"发现本地模型，从本地加载: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # 设置pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                print(f"从本地模型加载{self.model_size}基础模型成功")
                return
                
            except Exception as e:
                print(f"本地模型加载失败: {e}")
        
        # 第二步：检查HuggingFace缓存
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"检查HuggingFace缓存: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_hf,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print(f"从HuggingFace缓存加载{self.model_size}基础模型成功")
            
            # 保存到本地models目录
            print(f"保存模型到本地: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.base_model.save_pretrained(local_model_dir)
            print(f"模型已保存到: {local_model_dir}")
            
        except Exception as e:
            print(f"从HuggingFace缓存加载失败: {e}")
            
            # 第三步：从HuggingFace下载
            try:
                print(f"正在从HuggingFace下载{self.model_size}模型...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_hf,
                    force_download=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # 保存到本地models目录
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.base_model.save_pretrained(local_model_dir)
                print(f"{self.model_size}模型下载并保存到: {local_model_dir}")
                
            except Exception as e2:
                print(f"从HuggingFace下载也失败: {e2}")
                raise RuntimeError(f"无法加载{self.model_size}模型，所有方法都失败了")
    
    def _create_instruction_data(self, data: List[Tuple[str, int]]) -> Dataset:
        """创建指令格式的训练数据"""
        instructions = []
        
        for text, label in data:
            sentiment = "正面" if label == 1 else "负面"
            
            # 构建指令格式
            instruction = f"请分析以下微博文本的情感倾向，回答'正面'或'负面'。\n\n文本：{text}\n\n情感："
            response = sentiment
            
            
            # 组合成完整的训练文本
            full_text = f"{instruction}{response}{self.tokenizer.eos_token}"
            
            instructions.append({
                "instruction": instruction,
                "response": response,
                "text": full_text
            })
        
        return Dataset.from_list(instructions)
    
    def _tokenize_function(self, examples):
        """分词函数"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def _setup_lora(self, **kwargs):
        """设置LoRA配置"""
        lora_r = kwargs.get('lora_r', self.config['lora_r'])
        lora_alpha = kwargs.get('lora_alpha', self.config['lora_alpha'])
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        
        print(f"LoRA配置完成 (r={lora_r}, alpha={lora_alpha})")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")
        self.lora_model.print_trainable_parameters()  # PEFT库自带的参数统计
        
        return lora_config
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """训练模型"""
        print(f"开始训练 Qwen3-{self.model_size}-LoRA 模型...")
        
        # 加载基础模型
        self._load_base_model()
        
        # 设置LoRA
        self._setup_lora(**kwargs)
        
        # 超参数（使用配置文件的推荐值或用户指定值）
        num_epochs = kwargs.get('num_epochs', 3)
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'] // 2)  # LoRA需要更少批大小
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'] / 2)  # LoRA使用更小学习率
        output_dir = kwargs.get('output_dir', f'./models/qwen3_lora_{self.model_size.lower()}_checkpoints')
        
        print(f"超参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # 创建指令格式数据
        train_dataset = self._create_instruction_data(train_data)
        
        # 分词
        tokenized_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            report_to=None,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print(f"开始LoRA微调...")
        trainer.train()
        
        # 保存模型
        self.lora_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"Qwen3-{self.model_size}-LoRA 模型训练完成！")
    
    def _extract_sentiment(self, generated_text: str, instruction: str) -> int:
        """从生成的文本中提取情感标签"""
        response = generated_text[len(instruction):].strip()
        
        if "正面" in response:
            return 1
        elif "负面" in response:
            return 0
        else:
            return 0
    
    def predict(self, texts: List[str]) -> List[int]:
        """预测文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        predictions = []
        
        self.lora_model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc=f"Qwen3-{self.model_size}预测中"):
                pred, _ = self.predict_single(text)
                predictions.append(pred)
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """预测单条文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        # 构建指令
        instruction = f"请分析以下微博文本的情感倾向，回答'正面'或'负面'。\n\n文本：{text}\n\n情感："
        
        # 分词
        inputs = self.tokenizer(instruction, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回答
        self.lora_model.eval()
        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取情感标签
        prediction = self._extract_sentiment(generated_text, instruction)
        confidence = 0.8  # 生成式模型的置信度计算较复杂，这里给个固定值
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未训练")
        
        if model_path is None:
            model_path = MODEL_PATHS["lora"][self.model_size]
        
        os.makedirs(model_path, exist_ok=True)
        
        # 保存LoRA权重
        self.lora_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"LoRA模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载基础模型
        self._load_base_model()
        
        # 加载LoRA权重
        self.lora_model = PeftModel.from_pretrained(self.base_model, model_path)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"已加载Qwen3-{self.model_size}-LoRA模型: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qwen3-LoRA通用训练脚本')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='模型大小')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='测试数据路径')
    parser.add_argument('--model_path', type=str, help='模型保存路径（可选）')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批大小（可选，使用推荐值）')
    parser.add_argument('--learning_rate', type=float, help='学习率（可选，使用推荐值）')
    parser.add_argument('--lora_r', type=int, help='LoRA秩（可选，使用推荐值）')
    parser.add_argument('--max_samples', type=int, default=0, help='最大训练样本数（0表示使用全部数据）')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模式')
    
    args = parser.parse_args()
    
    # 如果没有指定模型大小，则询问用户
    if not args.model_size:
        print("Qwen3-LoRA模型训练")
        print("="*40)
        print("可用模型大小:")
        print("  1. 0.6B - 轻量级，训练快速，显存需求约8GB")
        print("  2. 4B  - 中等规模，性能均衡，显存需求约32GB") 
        print("  3. 8B  - 大规模，性能最佳，显存需求约64GB")
        print("\n注意: LoRA微调比Embedding方法需要更多显存")
        
        while True:
            choice = input("\n请选择模型大小 (1/2/3): ").strip()
            if choice == '1':
                args.model_size = '0.6B'
                break
            elif choice == '2':
                args.model_size = '4B'
                break
            elif choice == '3':
                args.model_size = '8B'
                break
            else:
                print("无效选择，请输入 1、2 或 3")
        
        print(f"已选择: Qwen3-{args.model_size} + LoRA")
        print()
    
    # 确保models目录存在
    os.makedirs('./models', exist_ok=True)
    
    # 创建模型
    model = Qwen3LoRAUniversal(args.model_size)
    
    # 确定模型保存路径
    model_path = args.model_path or MODEL_PATHS["lora"][args.model_size]
    
    if args.eval_only:
        # 仅评估模式
        print(f"评估模式：加载Qwen3-{args.model_size}-LoRA模型")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        # LoRA评估使用少量数据
        test_subset = test_data[:50]
        model.evaluate(test_subset)
    else:
        # 训练模式
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # 训练数据处理
        if args.max_samples > 0:
            train_subset = train_data[:args.max_samples]
            print(f"使用 {len(train_subset)} 条数据进行LoRA训练")
        else:
            train_subset = train_data
            print(f"使用全部 {len(train_subset)} 条数据进行LoRA训练")
        
        # 准备训练参数
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        if args.lora_r:
            train_kwargs['lora_r'] = args.lora_r
        
        # 训练模型
        model.train(train_subset, **train_kwargs)
        
        # 评估模型（使用少量测试数据）
        test_subset = test_data[:50]
        model.evaluate(test_subset)
        
        # 保存模型
        model.save_model(model_path)
        
        # 示例预测
        print(f"\nQwen3-{args.model_size}-LoRA 示例预测:")
        test_texts = [
            "今天天气真好，心情很棒",
            "这部电影太无聊了，浪费时间",
            "哈哈哈，太有趣了"
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "正面" if pred == 1 else "负面"
            print(f"文本: {text}")
            print(f"预测: {sentiment} (置信度: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()