import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import pandas as pd
import chardet  # 确保导入chardet

# 页面标题
st.title("📖 小说文本分析系统")

def detect_encoding(raw_data):
    """更健壮的编码检测函数"""
    try:
        detection = chardet.detect(raw_data)
        encoding = detection['encoding']
        confidence = detection['confidence']
        
        # 显示编码检测信息（用于调试）
        st.sidebar.write(f"检测到编码: {encoding} (置信度: {confidence:.2%})")
        
        # 备选编码优先级列表（根据中文文本调整）
        backup_encodings = ['utf-8', 'GB18030', 'GBK', 'GB2312', 'ISO-8859-1']
        
        # 置信度低于80%时尝试备选编码
        if confidence < 0.8:
            for enc in backup_encodings:
                try:
                    _ = raw_data.decode(enc)
                    st.sidebar.write(f"低置信度编码已验证可用: {enc}")
                    return enc
                except:
                    continue
        
        return encoding if encoding is not None else 'utf-8'
    except Exception as e:
        st.error(f"编码检测失败: {str(e)}")
        return 'utf-8'

def safe_decode(raw_data):
    """带错误处理的解码函数"""
    encoding = detect_encoding(raw_data)
    try:
        # 优先尝试严格模式解码
        return raw_data.decode(encoding)
    except UnicodeDecodeError:
        try:
            # 尝试用备选编码解码
            return raw_data.decode(encoding, errors='replace')
        except:
            # 最终回退方案
            return raw_data.decode('utf-8', errors='replace')

# 上传文件
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")

if uploaded_file:
    try:
        # 读取文件内容
        raw_data = uploaded_file.read()
        text = safe_decode(raw_data)  # 使用改进的解码方法

        # --- 以下为原有分析逻辑（保持不变） ---
        
        # 1. 分词处理
        words = jieba.lcut(text)
        word_freq = pd.Series(words).value_counts().head(20)
        
        # 显示词频表格
        st.subheader("📊 高频词汇统计")
        st.dataframe(word_freq)

        # 2. 生成词云
        st.subheader("☁ 词云图")
        wordcloud = WordCloud(
            font_path="SimHei.ttf", 
            background_color="white",
            width=800,  # 明确指定尺寸
            height=600
        ).generate(" ".join(words))
        
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

        # 3. LDA 主题分析（增加异常捕获）
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