import torch
import os
import logging
from LSTM_model import lstm_model_manager

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('lstm_predict')

class LSTMPredictor:
    """LSTM预测器，与当前系统的预测接口兼容"""
    
    def __init__(self):
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"初始化LSTM预测器，使用设备: {self.device}")
    
    def load_models(self, model_save_path, bert_model_path, tokenizer_path=None):
        """
        加载模型，与当前系统的model_manager.load_models接口兼容
        
        参数:
            model_save_path: LSTM模型保存路径
            bert_model_path: BERT模型路径
            tokenizer_path: 分词器路径（LSTM模型中使用BERT的分词器，可忽略）
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_save_path):
                logger.warning(f"模型文件 {model_save_path} 不存在，需要先训练模型")
                return False
                
            if not os.path.exists(bert_model_path):
                logger.error(f"BERT模型路径 {bert_model_path} 不存在")
                return False
            
            # 实际上我们在lstm_model_manager初始化时已经加载了模型，这里只是检查一下
            if lstm_model_manager.model is not None:
                self.model_loaded = True
                logger.info("LSTM模型已加载成功")
                return True
            else:
                logger.error("LSTM模型加载失败")
                return False
        except Exception as e:
            logger.error(f"加载模型过程中出错: {e}")
            return False
    
    def predict_batch(self, texts):
        """
        批量预测文本的情感
        
        参数:
            texts: 文本列表
            
        返回:
            predictions: 预测结果列表（0表示良好，1表示不良）
            probabilities: 预测概率列表
        """
        if not self.model_loaded and lstm_model_manager.model is None:
            logger.error("模型未加载，无法进行预测")
            return None, None
            
        if not texts:
            logger.warning("未提供文本，无法进行预测")
            return None, None
        
        try:
            # 调用LSTM模型管理器的批量预测函数
            predictions, probabilities = lstm_model_manager.predict_batch(texts)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"预测过程中出错: {e}")
            return None, None
    
    def predict(self, text):
        """
        预测单个文本的情感
        
        参数:
            text: 文本字符串
            
        返回:
            prediction: 预测结果（0表示良好，1表示不良）
            probability: 预测概率
        """
        if not self.model_loaded and lstm_model_manager.model is None:
            logger.error("模型未加载，无法进行预测")
            return None, None
            
        if not text or len(text.strip()) == 0:
            logger.warning("未提供文本或文本为空，无法进行预测")
            return None, None
        
        try:
            # 调用LSTM模型管理器的单个文本预测函数
            prediction, probability = lstm_model_manager.predict(text)
            return prediction, probability
        except Exception as e:
            logger.error(f"预测过程中出错: {e}")
            return None, None
    
    def train_model(self, train_texts, train_labels, val_texts=None, val_labels=None, 
                   batch_size=32, learning_rate=2e-5, epochs=10):
        """
        训练模型
        
        参数:
            train_texts: 训练集文本
            train_labels: 训练集标签
            val_texts: 验证集文本
            val_labels: 验证集标签
            batch_size: 批次大小
            learning_rate: 学习率
            epochs: 训练轮数
            
        返回:
            训练结果
        """
        try:
            results = lstm_model_manager.train(
                train_texts, train_labels, val_texts, val_labels, 
                batch_size, learning_rate, epochs
            )
            self.model_loaded = True
            return results
        except Exception as e:
            logger.error(f"训练模型过程中出错: {e}")
            return None

# 创建全局预测器实例
lstm_predictor = LSTMPredictor()

# 为了与现有代码兼容，提供一个与model_manager相同的predict_batch函数
def predict_batch(texts):
    return lstm_predictor.predict_batch(texts)

# 为了与现有代码兼容，提供一个与model_manager相同的load_models函数
def load_models(model_save_path, bert_model_path, tokenizer_path=None):
    return lstm_predictor.load_models(model_save_path, bert_model_path, tokenizer_path)

# 测试代码
if __name__ == "__main__":
    # 加载模型
    load_models(
        model_save_path="model_pro/lstm_model.pt",
        bert_model_path="model_pro/bert_model"
    )
    
    # 测试预测功能
    test_sentences = [
        "这件事情做得非常好",
        "服务太差了，态度恶劣",
        "这个产品质量一般，但价格便宜",
        "我对这家公司非常满意",
    ]
    
    for sentence in test_sentences:
        pred, prob = lstm_predictor.predict(sentence)
        if pred is not None:
            label = '良好' if pred == 0 else '不良'
            confidence = prob[pred]
            print(f"句子: '{sentence}' 预测结果: {label} (置信度: {confidence:.2%})")
        else:
            print(f"句子: '{sentence}' 预测失败") 