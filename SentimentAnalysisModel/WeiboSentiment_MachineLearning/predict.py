# -*- coding: utf-8 -*-
"""
统一的情感分析预测程序
支持加载所有模型进行情感预测
"""
import argparse
import os
import re
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 导入所有模型类
from bayes_train import BayesModel
from svm_train import SVMModel
from xgboost_train import XGBoostModel
from lstm_train import LSTMModel
from bert_train import BertModel_Custom
from utils import processing


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(self):
        self.models = {}
        self.available_models = {
            'bayes': BayesModel,
            'svm': SVMModel,
            'xgboost': XGBoostModel,
            'lstm': LSTMModel,
            'bert': BertModel_Custom
        }
        
    def load_model(self, model_type: str, model_path: str, **kwargs) -> None:
        """加载指定类型的模型
        
        Args:
            model_type: 模型类型 ('bayes', 'svm', 'xgboost', 'lstm', 'bert')
            model_path: 模型文件路径
            **kwargs: 其他参数（如BERT的预训练模型路径）
        """
        if model_type not in self.available_models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            return
        
        print(f"加载 {model_type.upper()} 模型...")
        
        try:
            if model_type == 'bert':
                # BERT需要额外的预训练模型路径
                bert_path = kwargs.get('bert_path', './model/chinese_wwm_pytorch')
                model = BertModel_Custom(bert_path)
            else:
                model = self.available_models[model_type]()
            
            model.load_model(model_path)
            self.models[model_type] = model
            print(f"{model_type.upper()} 模型加载成功")
            
        except Exception as e:
            print(f"加载 {model_type.upper()} 模型失败: {e}")
    
    def load_all_models(self, model_dir: str = './model', bert_path: str = './model/chinese_wwm_pytorch') -> None:
        """加载所有可用的模型
        
        Args:
            model_dir: 模型文件目录
            bert_path: BERT预训练模型路径
        """
        model_files = {
            'bayes': os.path.join(model_dir, 'bayes_model.pkl'),
            'svm': os.path.join(model_dir, 'svm_model.pkl'),
            'xgboost': os.path.join(model_dir, 'xgboost_model.pkl'),
            'lstm': os.path.join(model_dir, 'lstm_model.pth'),
            'bert': os.path.join(model_dir, 'bert_model.pth')
        }
        
        print("开始加载所有可用模型...")
        for model_type, model_path in model_files.items():
            self.load_model(model_type, model_path, bert_path=bert_path)
        
        print(f"\n已加载 {len(self.models)} 个模型: {list(self.models.keys())}")
    
    def predict_single(self, text: str, model_type: str = None) -> Dict[str, Tuple[int, float]]:
        """预测单条文本的情感
        
        Args:
            text: 待预测文本
            model_type: 指定模型类型，如果为None则使用所有已加载的模型
            
        Returns:
            Dict[model_type, (prediction, confidence)]
        """
        # 文本预处理
        processed_text = processing(text)
        
        if model_type:
            if model_type not in self.models:
                raise ValueError(f"模型 {model_type} 未加载")
            
            prediction, confidence = self.models[model_type].predict_single(processed_text)
            return {model_type: (prediction, confidence)}
        
        # 使用所有模型预测
        results = {}
        for name, model in self.models.items():
            try:
                prediction, confidence = model.predict_single(processed_text)
                results[name] = (prediction, confidence)
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                results[name] = (0, 0.0)
        
        return results
    
    def predict_batch(self, texts: List[str], model_type: str = None) -> Dict[str, List[int]]:
        """批量预测文本情感
        
        Args:
            texts: 待预测文本列表
            model_type: 指定模型类型，如果为None则使用所有已加载的模型
            
        Returns:
            Dict[model_type, predictions]
        """
        # 文本预处理
        processed_texts = [processing(text) for text in texts]
        
        if model_type:
            if model_type not in self.models:
                raise ValueError(f"模型 {model_type} 未加载")
            
            predictions = self.models[model_type].predict(processed_texts)
            return {model_type: predictions}
        
        # 使用所有模型预测
        results = {}
        for name, model in self.models.items():
            try:
                predictions = model.predict(processed_texts)
                results[name] = predictions
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                results[name] = [0] * len(texts)
        
        return results
    
    def ensemble_predict(self, text: str, weights: Dict[str, float] = None) -> Tuple[int, float]:
        """集成预测（多个模型投票）
        
        Args:
            text: 待预测文本
            weights: 模型权重，如果为None则平均权重
            
        Returns:
            (prediction, confidence)
        """
        if len(self.models) == 0:
            raise ValueError("没有加载任何模型")
        
        results = self.predict_single(text)
        
        if weights is None:
            weights = {name: 1.0 for name in results.keys()}
        
        # 加权平均
        total_weight = 0
        weighted_prob = 0
        
        for model_name, (pred, conf) in results.items():
            if model_name in weights:
                weight = weights[model_name]
                prob = conf if pred == 1 else 1 - conf
                weighted_prob += prob * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0, 0.5
        
        final_prob = weighted_prob / total_weight
        final_pred = int(final_prob > 0.5)
        final_conf = final_prob if final_pred == 1 else 1 - final_prob
        
        return final_pred, final_conf
    
    def interactive_predict(self):
        """交互式预测模式"""
        if len(self.models) == 0:
            print("错误: 没有加载任何模型，请先加载模型")
            return
        
        print("\n" + "="*50)
        print("="*50)
        print(f"已加载模型: {', '.join(self.models.keys())}")
        print("输入 'q' 退出程序")
        print("输入 'models' 查看模型列表")
        print("输入 'ensemble' 使用集成预测")
        print("-"*50)
        
        while True:
            try:
                text = input("\n请输入要分析的微博内容: ").strip()
                
                if text.lower() == 'q':
                    print("👋 再见！")
                    break
                
                if text.lower() == 'models':
                    print(f"已加载模型: {list(self.models.keys())}")
                    continue
                
                if text.lower() == 'ensemble':
                    if len(self.models) > 1:
                        pred, conf = self.ensemble_predict(text)
                        sentiment = "😊 正面" if pred == 1 else "😞 负面"
                        print(f"\n🤖 集成预测结果:")
                        print(f"   情感倾向: {sentiment}")
                        print(f"   置信度: {conf:.4f}")
                    else:
                        print("❌ 集成预测需要至少2个模型")
                    continue
                
                if not text:
                    print("❌ 请输入有效内容")
                    continue
                
                # 预测
                results = self.predict_single(text)
                
                print(f"\n📝 原文: {text}")
                print("🔍 预测结果:")
                
                for model_name, (pred, conf) in results.items():
                    sentiment = "😊 正面" if pred == 1 else "😞 负面"
                    print(f"   {model_name.upper():8}: {sentiment} (置信度: {conf:.4f})")
                
                # 如果有多个模型，显示集成结果
                if len(results) > 1:
                    ensemble_pred, ensemble_conf = self.ensemble_predict(text)
                    ensemble_sentiment = "😊 正面" if ensemble_pred == 1 else "😞 负面"
                    print(f"   {'集成':8}: {ensemble_sentiment} (置信度: {ensemble_conf:.4f})")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"❌ 预测过程中出现错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='微博情感分析统一预测程序')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='模型文件目录')
    parser.add_argument('--bert_path', type=str, default='./model/chinese_wwm_pytorch',
                        help='BERT预训练模型路径')
    parser.add_argument('--model_type', type=str, choices=['bayes', 'svm', 'xgboost', 'lstm', 'bert'],
                        help='指定单个模型类型进行预测')
    parser.add_argument('--text', type=str,
                        help='直接预测指定文本')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='交互式预测模式（默认）')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成预测')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = SentimentPredictor()
    
    # 加载模型
    if args.model_type:
        # 加载指定模型
        model_files = {
            'bayes': 'bayes_model.pkl',
            'svm': 'svm_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lstm': 'lstm_model.pth',
            'bert': 'bert_model.pth'
        }
        model_path = os.path.join(args.model_dir, model_files[args.model_type])
        predictor.load_model(args.model_type, model_path, bert_path=args.bert_path)
    else:
        # 加载所有模型
        predictor.load_all_models(args.model_dir, args.bert_path)
    
    # 如果指定了文本，直接预测
    if args.text:
        if args.ensemble and len(predictor.models) > 1:
            pred, conf = predictor.ensemble_predict(args.text)
            sentiment = "正面" if pred == 1 else "负面"
            print(f"文本: {args.text}")
            print(f"集成预测: {sentiment} (置信度: {conf:.4f})")
        else:
            results = predictor.predict_single(args.text, args.model_type)
            print(f"文本: {args.text}")
            for model_name, (pred, conf) in results.items():
                sentiment = "正面" if pred == 1 else "负面"
                print(f"{model_name.upper()}: {sentiment} (置信度: {conf:.4f})")
    elif args.interactive:
        # 交互式模式
        predictor.interactive_predict()


if __name__ == "__main__":
    main()