import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_default_wordcloud():
    """创建默认词云图片"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_dir = os.path.join(project_root, 'static')
    generated_dir = os.path.join(static_dir, 'generated')
    
    # 创建目录（如果不存在）
    os.makedirs(generated_dir, exist_ok=True)
    
    # 创建一个简单的词云图片
    words = {
        "微博": 100,
        "舆情": 80,
        "分析": 70,
        "系统": 60,
        "大数据": 50,
        "社交媒体": 40,
        "热点": 35,
        "话题": 30,
        "评论": 25,
        "转发": 20
    }
    
    wc = WordCloud(
        width=800,
        height=500,
        background_color='white',
        max_words=100
    )
    
    wc.generate_from_frequencies(words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # 保存默认词云图片
    default_path = os.path.join(static_dir, 'contentCloud.jpg')
    plt.savefig(default_path, format='jpg', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    
    print(f"默认词云图片已创建: {default_path}")

if __name__ == "__main__":
    create_default_wordcloud() 