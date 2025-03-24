import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from gensim import corpora, models
import pandas as pd
import chardet
import os

# ================== 初始化全局组件 ==================
st.title("📖 小说文本分析系统")
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")  # 必须位于函数定义之前

# ================== 工具函数定义 ==================
def find_chinese_font():
    """自动查找系统中可用的中文字体"""
    try:
        # 优先检查项目根目录字体
        font_candidates = [
            "SimHei.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/SimHei.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/Supplemental/Songti.ttc"
        ]
        
        for font in font_candidates:
            if os.path.exists(font):
                return font
        
        # 查找系统字体
        system_fonts = fm.findSystemFonts()
        for font in system_fonts:
            if any(keyword in font.lower() for keyword in ['hei', 'song', 'fang']):
                return font
        return None
    except Exception as e:
        st.error(f"字体查找失败: {str(e)}")
        return None

def safe_decode(raw_data):
    """安全解码实现"""
    try:
        detection = chardet.detect(raw_data)
        encoding = detection['encoding'] or 'utf-8'
        return raw_data.decode(encoding, errors='replace')
    except Exception as e:
        st.error(f"解码失败: {str(e)}")
        return raw_data.decode('utf-8', errors='replace')

# ================== 主处理逻辑 ==================
if uploaded_file is not None:  # 显式判断 None 更安全
    try:
        # 读取文件内容
        raw_data = uploaded_file.read()
        text = safe_decode(raw_data)

        # 1. 分词处理
        words = jieba.lcut(text)
        if len(words) == 0:
            raise ValueError("分词结果为空，请检查文件内容")
            
        # 2. 高频词统计
        st.subheader("📊 高频词汇统计")
        word_freq = pd.Series(words).value_counts().head(20)
        st.dataframe(word_freq)

        # 3. 生成词云
        st.subheader("☁ 词云图")
        font_path = find_chinese_font() or fm.findfont(fm.FontProperties(family='sans-serif'))
        
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color="white",
            collocations=False
        ).generate(" ".join(words))
        
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # 4. LDA分析
        st.subheader("🎯 主题关键词提取")
        try:
            dictionary = corpora.Dictionary([words])
            corpus = [dictionary.doc2bow(words)]
            lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary)
            for topic in lda.print_topics():
                st.code(f"主题 {topic[0]}: {topic[1]}")
        except Exception as e:
            st.error(f"主题分析失败: {str(e)}")

    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.stop()
else:
    st.info("👆 请先上传文本文件")