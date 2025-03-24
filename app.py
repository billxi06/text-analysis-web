import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 新增字体管理模块
from gensim import corpora, models
import pandas as pd
import chardet
import os

# 页面标题
st.title("📖 小说文本分析系统")

def find_chinese_font():
    """自动查找系统中可用的中文字体"""
    try:
        # 尝试加载项目目录下的字体文件
        if os.path.exists("SimHei.ttf"):
            return os.path.abspath("SimHei.ttf")
        
        # 查找系统已安装的中文字体
        font_paths = fm.findSystemFonts()
        chinese_fonts = [
            f for f in font_paths 
            if any(name in f.lower() for name in ['simhei', 'msyh', 'stheitisc', 'songti', 'fangsong'])
        ]
        
        # 常见系统字体路径（跨平台兼容）
        system_fonts = [
            # Windows
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            # MacOS
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            # Linux
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
        
        # 合并所有可能的字体路径
        all_fonts = chinese_fonts + system_fonts
        
        for font in all_fonts:
            if os.path.exists(font):
                return font
        return None
    except Exception as e:
        st.error(f"字体查找失败: {str(e)}")
        return None

def generate_wordcloud(text):
    """生成词云的健壮实现"""
    try:
        # 获取字体路径
        font_path = find_chinese_font()
        
        # 配置词云参数
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=600,
            collocations=False,  # 禁用双词模式
            max_words=200,
            margin=2
        )
        
        # 生成词云
        wordcloud = wc.generate(text)
        
        # 显示结果
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)
        
        # 显示使用的字体信息（调试用）
        if font_path:
            st.caption(f"使用的字体: {os.path.basename(font_path)}")
        else:
            st.warning("正在使用默认字体，中文显示可能异常")

    except Exception as e:
        st.error(f"生成词云失败: {str(e)}")
        st.markdown("""
            **常见解决方法：**
            1. 请确保项目包含 [SimHei.ttf](https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf) 字体文件
            2. 对于本地运行：将字体文件放在与代码相同的目录
            3. 对于云端部署：确保字体文件已上传到GitHub仓库
        """)

# ...（保持原有的 detect_encoding 和 safe_decode 函数不变）...

if uploaded_file:
    try:
        # 读取文件内容（保持不变）
        raw_data = uploaded_file.read()
        text = safe_decode(raw_data)

        # 1. 分词处理（保持不变）
        words = jieba.lcut(text)
        word_freq = pd.Series(words).value_counts().head(20)
        
        # 显示词频表格（保持不变）
        st.subheader("📊 高频词汇统计")
        st.dataframe(word_freq)

        # 2. 生成词云（使用新函数）
        st.subheader("☁ 词云图")
        generate_wordcloud(" ".join(words))  # 修改为调用新函数

        # ...（保持其他部分不变）...

    except Exception as e:
        st.error(f"处理文件时发生严重错误: {str(e)}")
        st.stop()