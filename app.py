import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from gensim import corpora, models
import pandas as pd
import chardet
import os

# 页面标题
st.title("📖 小说文本分析系统")

# ========== 核心修复点：确保上传组件在全局作用域 ==========
# 文件上传组件必须放在所有函数定义之外
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")

def find_chinese_font():
    """自动查找系统中可用的中文字体（保持不变）"""
    # ...（保持原有实现不变）...

def generate_wordcloud(text):
    """生成词云（保持不变）"""
    # ...（保持原有实现不变）...

def detect_encoding(raw_data):
    """编码检测（保持不变）"""
    # ...（保持原有实现不变）...

def safe_decode(raw_data):
    """安全解码（保持不变）"""
    # ...（保持原有实现不变）...

# ========== 处理逻辑必须与上传组件同级 ==========
if uploaded_file:
    try:
        # 读取文件内容
        raw_data = uploaded_file.read()
        text = safe_decode(raw_data)

        # 1. 分词处理
        words = jieba.lcut(text)
        word_freq = pd.Series(words).value_counts().head(20)
        
        # 显示词频表格
        st.subheader("📊 高频词汇统计")
        st.dataframe(word_freq)

        # 2. 生成词云
        st.subheader("☁ 词云图")
        generate_wordcloud(" ".join(words))

        # 3. LDA 主题分析
        st.subheader("🎯 主题关键词提取")
        try:
            texts = [words]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary)
            for topic in lda.print_topics():
                st.write(f"主题 {topic[0]}: {topic[1]}")
        except Exception as e:
            st.error(f"主题分析失败: {str(e)}")

        # 4. 人物关系分析（示例）
        st.subheader("🕸 人物关系网络（示例）")
        st.image("https://via.placeholder.com/600x400.png?text=人物关系图", use_column_width=True)

    except Exception as e:
        st.error(f"处理文件时发生严重错误: {str(e)}")
        st.stop()