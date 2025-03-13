import re
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger('sensitive_filter')
logger.setLevel(logging.INFO)

class SensitiveDataFilter:
    """
    敏感数据过滤器 - 用于检测和屏蔽输出内容中的敏感信息
    
    功能:
    1. 自动识别并过滤手机号、邮箱、身份证号、信用卡号等敏感信息
    2. 支持自定义敏感信息模式和替换文本
    3. 提供批量处理和实时过滤功能
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SensitiveDataFilter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 默认配置
        self.config = {
            'enabled': os.getenv('ENABLE_SENSITIVE_DATA_FILTER', 'true').lower() == 'true',
            'patterns': {
                'phone': r'\b1[3-9]\d{9}\b',
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'id_card': r'\b[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
                'credit_card': r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b',
                'address': r'(北京|上海|广州|深圳|天津|重庆|南京|杭州|武汉|成都|西安)市.*?(路|街|道|巷).*?(号)'
            },
            'replacements': {
                'phone': '***********',
                'email': '******@*****',
                'id_card': '******************',
                'credit_card': '****************',
                'address': '[地址已隐藏]'
            }
        }
        
        # 加载自定义配置
        self._load_config()
        
        # 编译正则表达式
        self._compile_patterns()
        
        self._initialized = True
        
        logger.info("敏感数据过滤器初始化完成")
        if self.config['enabled']:
            logger.info(f"已启用以下类型的敏感数据过滤: {', '.join(self.config['patterns'].keys())}")
        else:
            logger.info("敏感数据过滤器已禁用")
    
    def _load_config(self):
        """加载自定义配置"""
        # 配置文件路径
        data_dir = os.getenv('DATA_DIR', 'data')
        config_path = os.path.join(data_dir, 'security', 'sensitive_filter.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                
                # 更新配置
                if 'enabled' in custom_config:
                    self.config['enabled'] = custom_config['enabled']
                
                if 'patterns' in custom_config:
                    for key, pattern in custom_config['patterns'].items():
                        self.config['patterns'][key] = pattern
                
                if 'replacements' in custom_config:
                    for key, replacement in custom_config['replacements'].items():
                        self.config['replacements'][key] = replacement
                
                logger.info(f"已加载自定义敏感数据过滤配置: {config_path}")
            except Exception as e:
                logger.error(f"加载敏感数据过滤配置失败: {e}")
    
    def _compile_patterns(self):
        """编译正则表达式"""
        self.compiled_patterns = {}
        for key, pattern in self.config['patterns'].items():
            try:
                self.compiled_patterns[key] = re.compile(pattern)
                logger.debug(f"已编译敏感数据模式: {key} - {pattern}")
            except re.error as e:
                logger.error(f"编译敏感数据模式失败: {key} - {pattern}: {e}")
    
    def filter_text(self, text):
        """
        过滤文本中的敏感信息
        
        参数:
            text: 要过滤的文本
            
        返回:
            过滤后的文本
        """
        if not self.config['enabled'] or not text:
            return text
        
        filtered_text = text
        for key, pattern in self.compiled_patterns.items():
            replacement = self.config['replacements'].get(key, '[FILTERED]')
            filtered_text = pattern.sub(replacement, filtered_text)
        
        return filtered_text
    
    def filter_dict(self, data, *skip_keys):
        """
        过滤字典中的敏感信息
        
        参数:
            data: 要过滤的字典
            skip_keys: 要跳过的键（不进行过滤）
            
        返回:
            过滤后的字典
        """
        if not self.config['enabled'] or not data:
            return data
        
        if not isinstance(data, dict):
            if isinstance(data, str):
                return self.filter_text(data)
            return data
        
        filtered_data = {}
        for key, value in data.items():
            if key in skip_keys:
                filtered_data[key] = value
                continue
                
            if isinstance(value, dict):
                filtered_data[key] = self.filter_dict(value, *skip_keys)
            elif isinstance(value, list):
                filtered_data[key] = [
                    self.filter_dict(item, *skip_keys) if isinstance(item, (dict, list)) else
                    self.filter_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, str):
                filtered_data[key] = self.filter_text(value)
            else:
                filtered_data[key] = value
        
        return filtered_data
    
    def filter_list(self, data, *skip_keys):
        """
        过滤列表中的敏感信息
        
        参数:
            data: 要过滤的列表
            skip_keys: 如果列表项是字典，要跳过的键
            
        返回:
            过滤后的列表
        """
        if not self.config['enabled'] or not data:
            return data
        
        if not isinstance(data, list):
            if isinstance(data, dict):
                return self.filter_dict(data, *skip_keys)
            if isinstance(data, str):
                return self.filter_text(data)
            return data
        
        return [
            self.filter_dict(item, *skip_keys) if isinstance(item, dict) else
            self.filter_list(item, *skip_keys) if isinstance(item, list) else
            self.filter_text(item) if isinstance(item, str) else item
            for item in data
        ]
    
    def is_sensitive_info(self, text, info_type=None):
        """
        检查文本是否包含敏感信息
        
        参数:
            text: 要检查的文本
            info_type: 指定要检查的敏感信息类型，如果为None则检查所有类型
            
        返回:
            包含敏感信息返回True，否则返回False
        """
        if not self.config['enabled'] or not text:
            return False
        
        if info_type:
            if info_type not in self.compiled_patterns:
                logger.warning(f"未知的敏感信息类型: {info_type}")
                return False
            return bool(self.compiled_patterns[info_type].search(text))
        
        for pattern in self.compiled_patterns.values():
            if pattern.search(text):
                return True
        
        return False
    
    def get_sensitive_info_types(self, text):
        """
        获取文本中包含的敏感信息类型
        
        参数:
            text: 要检查的文本
            
        返回:
            包含的敏感信息类型列表
        """
        if not self.config['enabled'] or not text:
            return []
        
        types = []
        for key, pattern in self.compiled_patterns.items():
            if pattern.search(text):
                types.append(key)
        
        return types
    
    def enable(self):
        """启用敏感数据过滤器"""
        self.config['enabled'] = True
        logger.info("敏感数据过滤器已启用")
    
    def disable(self):
        """禁用敏感数据过滤器"""
        self.config['enabled'] = False
        logger.info("敏感数据过滤器已禁用")
    
    def is_enabled(self):
        """检查敏感数据过滤器是否启用"""
        return self.config['enabled']
    
    def add_pattern(self, key, pattern, replacement='[FILTERED]'):
        """
        添加自定义敏感信息模式
        
        参数:
            key: 敏感信息类型标识
            pattern: 正则表达式字符串
            replacement: 替换文本
        """
        try:
            # 测试是否是有效的正则表达式
            re.compile(pattern)
            
            # 更新配置
            self.config['patterns'][key] = pattern
            self.config['replacements'][key] = replacement
            
            # 重新编译正则表达式
            self._compile_patterns()
            
            logger.info(f"已添加敏感信息模式: {key}")
            return True
        except re.error as e:
            logger.error(f"添加敏感信息模式失败: {key} - {pattern}: {e}")
            return False
    
    def remove_pattern(self, key):
        """
        移除敏感信息模式
        
        参数:
            key: 敏感信息类型标识
        """
        if key in self.config['patterns']:
            del self.config['patterns'][key]
            
            if key in self.config['replacements']:
                del self.config['replacements'][key]
            
            if key in self.compiled_patterns:
                del self.compiled_patterns[key]
            
            logger.info(f"已移除敏感信息模式: {key}")
            return True
        
        logger.warning(f"未找到敏感信息模式: {key}")
        return False
    
    def save_config(self):
        """保存当前配置到文件"""
        data_dir = os.getenv('DATA_DIR', 'data')
        security_dir = os.path.join(data_dir, 'security')
        os.makedirs(security_dir, exist_ok=True)
        
        config_path = os.path.join(security_dir, 'sensitive_filter.json')
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"敏感数据过滤配置已保存到: {config_path}")
            return True
        except Exception as e:
            logger.error(f"保存敏感数据过滤配置失败: {e}")
            return False

# 创建全局敏感数据过滤器实例
sensitive_filter = SensitiveDataFilter()

# 提供便捷的过滤函数
def filter_text(text):
    """过滤文本中的敏感信息"""
    return sensitive_filter.filter_text(text)

def filter_dict(data, *skip_keys):
    """过滤字典中的敏感信息"""
    return sensitive_filter.filter_dict(data, *skip_keys)

def filter_list(data, *skip_keys):
    """过滤列表中的敏感信息"""
    return sensitive_filter.filter_list(data, *skip_keys)

def is_sensitive_info(text, info_type=None):
    """检查文本是否包含敏感信息"""
    return sensitive_filter.is_sensitive_info(text, info_type)

# 示例用法
if __name__ == "__main__":
    # 测试文本
    test_text = """
    联系人: 张三
    电话: 13812345678
    邮箱: zhangsan@example.com
    身份证: 110101199001011234
    地址: 北京市海淀区中关村大街20号
    信用卡: 6225 1234 5678 9012
    """
    
    # 过滤敏感信息
    filtered_text = filter_text(test_text)
    print("原始文本:")
    print(test_text)
    print("\n过滤后:")
    print(filtered_text)
    
    # 检查敏感信息类型
    types = sensitive_filter.get_sensitive_info_types(test_text)
    print(f"\n包含的敏感信息类型: {types}") 