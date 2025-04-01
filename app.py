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
import plotly.graph_objects as go

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
st.title("📖 文本分析系统")
uploaded_file = st.file_uploader("上传文本文件（.txt）", type="txt")

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

def analyze_relationships(text, target_character, top_n=15):
    """改进版关系分析函数"""
    jieba.add_word(target_character)
    relationships = defaultdict(int)
    window_size = 8  # 扩大上下文窗口
    
    words = jieba.lcut(text)
    for i in range(len(words)):
        if words[i] == target_character:
            # 动态窗口调整
            start = max(0, i - window_size)
            end = min(len(words), i + window_size)
            neighbors = words[start:end]
            
            # 增强过滤条件
            for char in neighbors:
                if (char != target_character and 
                    len(char) > 1 and 
                    not char.isdigit() and
                    not any(c in char for c in ['，','。','！'])):
                    relationships[char] += 1
    
    # 按权重排序并截取
    sorted_rels = sorted(relationships.items(), 
                        key=lambda x: x[1], 
                        reverse=True)[:top_n]
    return dict(sorted_rels)

def plot_static_network(target_character, relationships):
    """优化静态布局"""
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    G = nx.Graph()
    G.add_node(target_character)
    
    for char, weight in relationships.items():
        G.add_node(char)
        G.add_edge(target_character, char, weight=weight)
    
    # 改进布局参数
    pos = nx.spring_layout(G, 
                          k=0.3, 
                          iterations=100, 
                          seed=42)
    
    plt.figure(figsize=(12,10))
    
    # 动态调整节点大小
    node_sizes = [2500 if n == target_character else 1200 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=['gold' if n == target_character else 'lightblue' for n in G.nodes()],
                          alpha=0.9)
    
    # 动态调整边宽
    edge_widths = [d['weight']*0.5 for u,v,d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, 
                          width=edge_widths,
                          edge_color='gray',
                          alpha=0.6)
    
    # 优化标签显示
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=12,
                           font_family='SimHei')
    
    plt.title(f"'{target_character}' 人物关系网络", fontsize=16)
    plt.axis("off")
    st.pyplot(plt.gcf())

def plot_interactive_network(target_character, relationships):
    """交互式可视化"""
    G = nx.Graph()
    G.add_node(target_character)
    
    for char, weight in relationships.items():
        G.add_node(char)
        G.add_edge(target_character, char, weight=weight)
    
    # 生成布局
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # 创建边轨迹
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # 创建节点轨迹
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(25 if node != target_character else 40)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=[],
            line_width=2,
            opacity=0.9))
    
    # 添加交互功能
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    clickmode='select+event',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=800
                    ))
    
    # 添加点击交互
    fig.update_traces(
        textfont=dict(family='SimHei', size=14),
        selector=dict(type='scatter', mode='markers+text')
    )
    st.plotly_chart(fig, use_container_width=True)
    
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

        # 改进的人物关系分析模块
        st.subheader("🕸️ 人物关系网络分析")
        target_character = st.text_input("输入要分析的人物名称（如：段誉）:")
        
        if target_character:
            if target_character not in text:
                st.warning("⚠️ 该人物不存在于文本中")
            else:
                # 添加控制面板
                col1, col2 = st.columns(2)
                with col1:
                    top_n = st.slider("选择显示关联人数", 5, 50, 15, 
                                    help="根据文本复杂度调整显示人数")
                with col2:
                    layout_type = st.selectbox("选择可视化类型", 
                                             ["交互式布局", "静态布局"],
                                             index=0)
                
                relationships = analyze_relationships(text, target_character, top_n)
                
                if not relationships:
                    st.warning("未找到相关人物关系")
                else:
                    if layout_type == "静态布局":
                        plot_static_network(target_character, relationships)
                    else:
                        plot_interactive_network(target_character, relationships)
                    
                    # 显示统计信息
                    st.success(f"成功分析到 {len(relationships)} 个关键关系")
                    with st.expander("查看详细关系数据"):
                        st.write(pd.DataFrame({
                            "关联人物": relationships.keys(),
                            "关联强度": relationships.values()
                        }).sort_values("关联强度", ascending=False))

    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.stop()
else:
    st.info("👆 请先上传文本文件")