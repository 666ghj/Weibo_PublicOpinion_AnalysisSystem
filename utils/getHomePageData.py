from utils.getPublicData import getAllCommentsData,getAllArticleData
from datetime import datetime
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.query import query
from utils.logger import app_logger as logging

commentsList = getAllCommentsData()
articleList = getAllArticleData()

def getHomeTagsData():
    """获取首页标签数据，确保即使无数据也返回有效格式"""
    try:
        # 获取文章数量
        sql = "SELECT COUNT(*) as articleCount FROM article"
        data = query(sql, query_type="select")
        if not data or len(data) == 0:
            articleLenMax = 0
        else:
            articleLenMax = data[0]['articleCount'] if data[0]['articleCount'] else 0
            
        # 获取点赞数最高的作者名
        sql = "SELECT authorName FROM article WHERE likeNum = (SELECT MAX(likeNum) FROM article)"
        data = query(sql, query_type="select")
        if not data or len(data) == 0:
            likeCountMaxAuthorName = "无数据"
        else:
            likeCountMaxAuthorName = data[0]['authorName'] if data[0]['authorName'] else "无数据"
            
        # 获取评论最多的城市
        sql = "SELECT region, COUNT(*) as count FROM comments GROUP BY region ORDER BY count DESC LIMIT 1"
        data = query(sql, query_type="select")
        if not data or len(data) == 0:
            cityMax = "无数据"
        else:
            cityMax = data[0]['region'] if data[0]['region'] else "无数据"
            
        return articleLenMax, likeCountMaxAuthorName, cityMax
    except Exception as e:
        logging.error(f"获取首页标签数据出错: {e}")
        return "无数据", "无数据", "无数据"

def getHomeCommentsLikeCountTopFore():
    """获取评论点赞Top4，确保返回有效列表"""
    try:
        sql = "SELECT * FROM comments ORDER BY likes_counts DESC LIMIT 4"
        data = query(sql, query_type="select")
        return data if data else []
    except Exception as e:
        logging.error(f"获取评论点赞Top4数据出错: {e}")
        return []

def getHomeArticleCreatedAtChart():
    """
    获取首页文章发布时间统计图表数据
    返回：X轴日期列表，Y轴每个日期对应的文章数量
    """
    try:
        # 获取文章发布日期统计
        sql = """
        SELECT 
            SUBSTRING(created_at, 1, 10) as pub_date,
            COUNT(*) as article_count
        FROM 
            article
        WHERE 
            created_at IS NOT NULL
        GROUP BY 
            SUBSTRING(created_at, 1, 10)
        ORDER BY 
            pub_date DESC
        LIMIT 30
        """
        
        result = query(sql, query_type="select")
        
        # 检查结果是否为空
        if not result:
            logging.warning("未找到文章发布日期数据")
            return [], []
            
        # 反转结果以获得按日期升序的数据
        result = result[::-1]
        
        # 提取日期和数量
        X = [row['pub_date'] if row['pub_date'] else '未知日期' for row in result]
        Y = [row['article_count'] for row in result]
        
        logging.info(f"获取到{len(X)}个日期的文章统计数据")
        return X, Y
    except Exception as e:
        logging.error(f"获取文章发布时间图表数据时出错: {e}")
        return [], []

def getHomeTypeChart():# 统计每种类型的文章数量
    try:
        sql = "SELECT type, COUNT(*) as count FROM article GROUP BY type"
        data = query(sql, query_type="select")
        if not data:
            return {}
        
        result = [{'name': item['type'], 'value': item['count']} for item in data]
        return result
    except Exception as e:
        logging.error(f"获取类型图表数据出错: {e}")
        return {}

def getHomeCommentCreatedChart():# 统计每天用户评论数量
    try:
        sql = """
        SELECT DATE_FORMAT(STR_TO_DATE(created_at, '%a %b %d %H:%i:%s +0800 %Y'), '%Y-%m-%d') as date, 
               COUNT(*) as count 
        FROM comments 
        GROUP BY date 
        ORDER BY date
        """
        data = query(sql, query_type="select")
        if not data:
            return {}
        
        result = [{'name': item['date'], 'value': item['count']} for item in data]
        return result
    except Exception as e:
        logging.error(f"获取评论创建图表数据出错: {e}")
        return {}

def stopWordList():
    """获取停用词列表，如果文件不存在则返回默认列表"""
    try:
        return [line.strip() for line in open('./stopWords.txt', encoding='utf8').readlines()]
    except FileNotFoundError:
        logging.warning("停用词文件不存在，使用默认停用词列表")
        # 返回常用中文停用词
        return ['的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有', 
                '我们', '你们', '他们', '它们', '啊', '吧', '呢', '哦', '哈', '呀', '么', '要', 
                '这', '那', '你', '我', '他', '她', '它', '但是', '因为', '所以', '如果', '只是',
                '然而', '并且', '当然', '虽然', '不过', '这样', '这么', '那么', '如此', '因此',
                '然后', '接着', '随后', '首先', '其次', '再次', '最后', '总之', '一般', '通常',
                '可以', '可能', '应该', '似乎', '好像', '仿佛', '大约', '或者', '一些', '许多',
                '很多', '几个', '少数', '部分', '全部', '所有', '每个', '有些', '自己', '其他',
                '也', '还', '又', '再', '已', '才', '刚', '曾', '将', '会', '能', '应', '该',
                '于', '在', '由', '从', '把', '被', '给', '对', '向', '让', '使', '得', '地', '的',
                '得', '着', '过', '了', '个', '一', '两', '三', '四', '五', '六', '七', '八',
                '九', '十', '百', '千', '万', '亿', '多', '少', '大', '小', '高', '低', '中',
                '上', '下', '前', '后', '左', '右', '内', '外', '里', '外', '来', '去', '进',
                '出', '回', '到', '了', '啊', '哦', '呵', '嗯', '哼', '哈', '呀', '喂']

def getUserNameWordCloud():
    """生成用户名词云"""
    try:
        import os
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        text = ''
        stopWords = stopWordList()
        
        # 获取评论用户名
        sql = """
        SELECT DISTINCT authorName 
        FROM comments 
        WHERE authorName IS NOT NULL AND authorName != ''
        LIMIT 2000
        """
        
        results = query(sql, query_type="select")
        
        if not results:
            logging.warning("没有找到用户名数据，无法生成词云")
            return False
            
        for row in results:
            if 'authorName' in row and row['authorName']:
                text += row['authorName'] + ' '
        
        if not text:
            logging.warning("提取的用户名内容为空，无法生成词云")
            return False
            
        cut = jieba.cut(text)
        newCut = []
        for word in cut:
            if word not in stopWords and len(word) > 1:
                newCut.append(word)
                
        if not newCut:
            logging.warning("分词后内容为空，无法生成词云")
            return False
            
        # 尝试几个常见的中文字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/AppleGothic.ttf',  # 添加更多字体选项
            '/System/Library/Fonts/Apple LiGothic Medium.ttf',
            '/System/Library/Assets/com_apple_MobileAsset_Font_CJK/67c51367188e14be74f46dfd2552fb18be9dccef.asset/AssetData/Library/Fonts/ヒラギノ角ゴシック W3.ttc', # 苹果系统上可能有的日语字体含有中文字符
            '/System/Library/Fonts/PingFang.ttc',
            '/Users/auroral/Library/Fonts/SimHei.ttf',  # 用户可能自己安装的字体
            '/Users/auroral/Library/Fonts/SimSun.ttc',
            '/Users/auroral/Library/Fonts/NotoSansCJK-Regular.ttc',
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                logging.info(f"找到可用字体: {font_path}")
                break
        
        # 如果没找到任何字体，使用matplotlib默认字体
        if not font_path:
            logging.warning("未找到中文字体，将使用默认字体")
        
        string = ' '.join(newCut)
        wc_params = {
            'width': 1000,
            'height': 600,
            'background_color': '#fff',
            'colormap': 'Blues',
            'max_words': 200,
            'prefer_horizontal': 0.9,
            'scale': 2,
            'regexp': r"[\w\u4e00-\u9fff]+",  # 匹配中文和英文单词
            'collocations': False  # 不包含词组
        }
        
        # 如果找到了字体，添加到参数中
        if font_path:
            wc_params['font_path'] = font_path
            
        wc = WordCloud(**wc_params)
        wc.generate_from_text(string)
        
        # 确保静态目录存在
        static_dir = '/Users/auroral/ProjectDevelopment/Weibo_PublicOpinion_AnalysisSystem/static'
        os.makedirs(static_dir, exist_ok=True)
        
        image_path = os.path.join(static_dir, 'authorNameCloud.jpg')
        plt.figure(figsize=(10, 6))
        plt.imshow(wc)
        plt.axis('off')
        plt.savefig(image_path, dpi=300)
        plt.close()
        logging.info(f"成功生成用户名词云: {image_path}")
        return True
    except Exception as e:
        logging.error(f"生成用户名词云出错: {str(e)}", exc_info=True)
        return False

def getCommentWordCloud():
    """生成评论内容词云"""
    try:
        import os
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        text = ''
        stopWords = stopWordList()
        
        # 从数据库直接获取评论内容
        sql = """
        SELECT content 
        FROM comments 
        WHERE content IS NOT NULL AND content != '' 
        LIMIT 1000
        """
        
        results = query(sql, query_type="select")
        
        if not results:
            logging.warning("没有找到评论内容，无法生成词云")
            return False
            
        for row in results:
            if 'content' in row and row['content']:
                text += row['content'] + ' '
        
        if not text:
            logging.warning("提取的评论内容为空，无法生成词云")
            return False
            
        cut = jieba.cut(text)
        newCut = []
        for word in cut:
            if word not in stopWords and len(word) > 1:
                newCut.append(word)
                
        if not newCut:
            logging.warning("分词后内容为空，无法生成词云")
            return False
            
        # 尝试几个常见的中文字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/Apple LiGothic Medium.ttf',
            '/System/Library/Assets/com_apple_MobileAsset_Font_CJK/67c51367188e14be74f46dfd2552fb18be9dccef.asset/AssetData/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/Users/auroral/Library/Fonts/SimHei.ttf',
            '/Users/auroral/Library/Fonts/SimSun.ttc',
            '/Users/auroral/Library/Fonts/NotoSansCJK-Regular.ttc',
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                logging.info(f"找到可用字体: {font_path}")
                break
        
        # 如果没找到任何字体，使用matplotlib默认字体
        if not font_path:
            logging.warning("未找到中文字体，将使用默认字体")
        
        string = ' '.join(newCut)
        wc_params = {
            'width': 1000,
            'height': 600,
            'background_color': '#fff',
            'colormap': 'Reds',  # 使用不同于用户名词云的颜色图
            'max_words': 200,
            'prefer_horizontal': 0.9,
            'scale': 2,
            'regexp': r"[\w\u4e00-\u9fff]+",  # 匹配中文和英文单词
            'collocations': False  # 不包含词组
        }
        
        # 如果找到了字体，添加到参数中
        if font_path:
            wc_params['font_path'] = font_path
            
        wc = WordCloud(**wc_params)
        wc.generate_from_text(string)
        
        # 确保静态目录存在
        static_dir = '/Users/auroral/ProjectDevelopment/Weibo_PublicOpinion_AnalysisSystem/static'
        os.makedirs(static_dir, exist_ok=True)
        
        image_path = os.path.join(static_dir, 'commentCloud.jpg')
        plt.figure(figsize=(10, 6))
        plt.imshow(wc)
        plt.axis('off')
        plt.savefig(image_path, dpi=300)
        plt.close()
        logging.info(f"成功生成评论词云: {image_path}")
        return True
    except Exception as e:
        logging.error(f"生成评论词云出错: {str(e)}", exc_info=True)
        return False

def ensure_default_wordclouds():
    """确保默认词云文件存在"""
    try:
        import os
        static_dir = '/Users/auroral/ProjectDevelopment/Weibo_PublicOpinion_AnalysisSystem/static'
        os.makedirs(static_dir, exist_ok=True)
        
        # 检查并创建默认的用户名词云
        author_cloud_path = os.path.join(static_dir, 'authorNameCloud.jpg')
        if not os.path.exists(author_cloud_path):
            create_empty_wordcloud(author_cloud_path, "User Names")
        
        # 检查并创建默认的评论词云
        comment_cloud_path = os.path.join(static_dir, 'commentCloud.jpg')
        if not os.path.exists(comment_cloud_path):
            create_empty_wordcloud(comment_cloud_path, "Comments")
                
        return True
    except Exception as e:
        logging.error(f"确保默认词云文件存在时出错: {e}")
        return False

def create_empty_wordcloud(filepath, title="WordCloud"):
    """创建一个带有文字的空白图片作为默认词云"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        # 使用英文提示避免中文字体问题
        plt.text(0.5, 0.5, f"No data available for {title}", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=30)
        plt.axis('off')
        plt.savefig(filepath, dpi=100)
        plt.close()
        logging.info(f"创建默认词云文件: {filepath}")
        return True
    except Exception as e:
        logging.error(f"创建默认词云文件时出错: {e}")
        return False

# 应用启动时确保默认词云存在
ensure_default_wordclouds()

