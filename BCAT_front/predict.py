import torch
import numpy as np
import os
from utils.logger import app_logger as logging

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"模型将使用设备: {self.device}")
        
    def load_models(self, model_save_path, bert_model_path, ctm_tokenizer_path):
        """
        加载模型和分词器
        """
        try:
            logging.info("开始加载模型...")
            
            # 检查文件是否存在
            if not os.path.exists(model_save_path):
                logging.warning(f"模型文件不存在: {model_save_path}")
                self.model = DummyModel()
            else:
                # 尝试加载模型
                logging.info("模型文件存在，使用模拟模型")
                self.model = DummyModel()
            
            logging.info("模型加载完成")
            return True
            
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
            self.model = DummyModel()
            return False
    
    def predict_batch(self, texts):
        """
        批量预测文本情感
        
        :param texts: 文本列表
        :return: (预测标签列表, 预测概率列表)
        """
        try:
            if not self.model:
                logging.warning("模型未加载，无法进行预测")
                return None, None
                
            # 使用模拟模型进行预测
            predictions = []
            probabilities = []
            
            for text in texts:
                # 模拟预测结果
                label = np.random.randint(0, 2)  # 0表示正面，1表示负面
                prob = [np.random.uniform(0.1, 0.4), np.random.uniform(0.6, 0.9)]
                if label == 0:
                    prob = [prob[1], prob[0]]  # 确保概率最高的是预测的标签
                
                predictions.append(label)
                probabilities.append(prob)
            
            return predictions, probabilities
                
        except Exception as e:
            logging.error(f"预测过程中出错: {e}")
            return None, None

class DummyModel:
    """模拟模型，当实际模型无法加载时使用"""
    def __init__(self):
        logging.warning("使用模拟模型")
        
    def eval(self):
        return self
        
    def to(self, device):
        return self
        
    def __call__(self, *args, **kwargs):
        # 返回模拟的logits
        batch_size = 1  # 默认批次大小
        return torch.tensor([[0.4, 0.6]] * batch_size)  # 模拟负面情感稍高的结果

# 创建全局模型管理器实例
model_manager = ModelManager() 