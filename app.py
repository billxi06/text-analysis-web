import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import pandas as pd
import chardet

# 页面标题
st.title("📖 小说文本分析系统")

# 上传文件
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")

    

if uploaded_file:
     # 读取文件内容
    raw_data = uploaded_file.read()
    encoding = chardet.detect(raw_data)["encoding"]  # 自动检测编码
    text = raw_data.decode(encoding)                 # 使用检测到的编码解码

    # 1. 分词处理
    words = jieba.lcut(text)
    word_freq = pd.Series(words).value_counts().head(20)
    
    # 显示词频表格
    st.subheader("📊 高频词汇统计")
    st.dataframe(word_freq)

    # 2. 生成词云
    st.subheader("☁️ 词云图")
    wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white").generate(" ".join(words))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)

    # 3. LDA 主题分析（简化版）
    st.subheader("🎯 主题关键词提取")
    texts = [words]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary)
    for topic in lda.print_topics():
        st.write(f"主题 {topic[0]}: {topic[1]}")

    # 4. 人物关系分析（示例）
    st.subheader("🕸️ 人物关系网络（示例）")
    st.image("https://via.placeholder.com/600x400.png?text=人物关系图", use_column_width=True)