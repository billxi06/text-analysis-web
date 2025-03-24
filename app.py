import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import pandas as pd
import chardet
import os
import requests
import tempfile

# 缓存字体下载函数
@st.cache_resource
def download_font():
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    try:
        response = requests.get(font_url)
        response.raise_for_status()
        # 创建临时文件保存字体
        temp_font = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        temp_font.write(response.content)
        temp_font.close()
        return temp_font.name
    except Exception as e:
        st.error(f"字体下载失败: {str(e)}")
        return None

# ================== 初始化全局组件 ==================
st.title("📖 小说文本分析系统")
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")

# ================== 工具函数定义 ==================
def find_chinese_font():
    """强制使用本地字体文件（终极解决方案）"""
    font_path = download_font()
    if font_path:
        return os.path.abspath(font_path)
    st.error("❌ 无法获取中文字体文件")
    st.stop()

def generate_wordcloud(text):
    """完全依赖本地字体的词云生成函数"""
    try:
        # 获取绝对字体路径
        font_path = find_chinese_font()

        # 创建词云对象（强制使用指定字体）
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color="white",
            collocations=False,
            max_words=200,
            prefer_horizontal=0.9  # 优化中文显示
        )

        # 生成并显示词云
        wordcloud = wc.generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

        # 显示调试信息
        st.caption(f"当前使用字体路径: {font_path}")

    except Exception as e:
        st.error(f"词云生成失败: {str(e)}")
        st.image("https://via.placeholder.com/800x600.png?text=词云生成失败",
                 use_column_width=True,
                 caption="生成失败示例图")

def safe_decode(raw_data):
    """安全解码实现（保持原样）"""
    try:
        detection = chardet.detect(raw_data)
        encoding = detection['encoding'] or 'utf-8'
        return raw_data.decode(encoding, errors='replace')
    except Exception as e:
        st.error(f"解码失败: {str(e)}")
        return raw_data.decode('utf-8', errors='replace')

# ================== 主处理逻辑 ==================
if uploaded_file is not None:
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

        # 3. 生成词云（强制使用本地字体方案）
        st.subheader("☁ 词云图")
        generate_wordcloud(" ".join(words))

        # 4. LDA分析（保持原样）
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
    