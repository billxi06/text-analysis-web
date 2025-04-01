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
import networkx as nx
from collections import defaultdict

# 缓存字体下载函数
@st.cache_resource
def download_font():
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    try:
        response = requests.get(font_url)
        response.raise_for_status()
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
    font_path = download_font()
    if font_path:
        return os.path.abspath(font_path)
    st.error("❌ 无法获取中文字体文件")
    st.stop()

def generate_wordcloud(text):
    try:
        font_path = find_chinese_font()
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color="white",
            collocations=False,
            max_words=200,
            prefer_horizontal=0.9
        )
        wordcloud = wc.generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"词云生成失败: {str(e)}")

def safe_decode(raw_data):
    try:
        detection = chardet.detect(raw_data)
        encoding = detection['encoding'] or 'utf-8'
        return raw_data.decode(encoding, errors='replace')
    except Exception as e:
        return raw_data.decode('utf-8', errors='replace')

# 新增人物关系分析函数
def analyze_relationships(text, target_character):
    # 添加自定义词典提高人物识别
    jieba.add_word(target_character)
    
    # 构建共现关系
    relationships = defaultdict(int)
    window_size = 5  # 共现窗口大小
    
    words = jieba.lcut(text)
    for i in range(len(words)):
        if words[i] == target_character:
            start = max(0, i - window_size)
            end = min(len(words), i + window_size)
            neighbors = words[start:end]
            for char in neighbors:
                if char != target_character and len(char) > 1:  # 过滤单字和非目标人物
                    relationships[char] += 1
    
    return dict(relationships)

def plot_network(target_character, relationships):
    G = nx.Graph()
    G.add_node(target_character)
    
    for char, weight in relationships.items():
        G.add_node(char)
        G.add_edge(target_character, char, weight=weight)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10,8))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_family='SimHei', font_size=12)
    
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(f"'{target_character}' 人物关系网络")
    plt.axis("off")
    st.pyplot(plt)

# ================== 主处理逻辑 ==================
if uploaded_file is not None:
    try:
        raw_data = uploaded_file.read()
        text = safe_decode(raw_data)
        words = jieba.lcut(text)

        # 高频词统计
        st.subheader("📊 高频词汇统计")
        word_freq = pd.Series(words).value_counts().head(20)
        st.dataframe(word_freq)

        # 词云生成
        st.subheader("☁ 词云图")
        generate_wordcloud(" ".join(words))

        # LDA分析
        st.subheader("🎯 主题关键词提取")
        try:
            dictionary = corpora.Dictionary([words])
            corpus = [dictionary.doc2bow(words)]
            lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary)
            for topic in lda.print_topics():
                st.code(f"主题 {topic[0]}: {topic[1]}")
        except Exception as e:
            st.error(f"主题分析失败: {str(e)}")

        # 新增人物关系分析模块
        st.subheader("🕸️ 人物关系网络分析")
        target_character = st.text_input("输入要分析的人物名称（如：段誉）:")
        if target_character:
            if target_character not in text:
                st.warning("⚠️ 该人物不存在于文本中")
            else:
                relationships = analyze_relationships(text, target_character)
                if not relationships:
                    st.warning("未找到相关人物关系")
                else:
                    plot_network(target_character, relationships)
                    st.info(f"共找到 {len(relationships)} 个关联人物")

    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.stop()
else:
    st.info("👆 请先上传文本文件")