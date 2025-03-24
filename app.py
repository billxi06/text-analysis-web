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
uploaded_file = st.file_uploader("上传小说文本文件（.txt）", type="txt")

# ================== 工具函数定义 ==================
def find_chinese_font():
    """更安全的字体查找函数（带四级回退机制）"""
    try:
        # 1. 优先检查项目目录字体
        if os.path.exists("SimHei.ttf"):
            return os.path.abspath("SimHei.ttf")
        
        # 2. 查找系统预置路径
        font_paths = [
            # Windows
            'C:\\Windows\\Fonts\\msyh.ttc',
            'C:\\Windows\\Fonts\\simhei.ttf',
            # MacOS
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            # Linux
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            # 备用路径
            '/usr/share/fonts/truetype/msttcorefonts/SimHei.ttf'
        ]
        
        # 3. 验证预置路径有效性
        valid_fonts = [f for f in font_paths if os.path.exists(f)]
        if valid_fonts:
            return valid_fonts[0]
        
        # 4. 使用matplotlib默认中文字体
        try:
            from matplotlib import rcParams
            default_font = rcParams['font.sans-serif'][0]
            if isinstance(default_font, str):
                return default_font
            return fm.findfont(default_font)
        except:
            return None
            
    except Exception as e:
        st.error(f"字体查找失败: {str(e)}")
        return None

def generate_wordcloud(text):
    """完全重写的词云生成函数"""
    try:
        # 获取字体路径（带安全验证）
        font_path = find_chinese_font()
        
        # 验证字体路径有效性
        if font_path and not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件不存在: {font_path}")
        
        # 创建词云对象
        wc = WordCloud(
            font_path=font_path if font_path else None,
            width=800,
            height=600,
            background_color="white",
            collocations=False,
            max_words=200,
            margin=2,
            prefer_horizontal=0.9  # 优化中文横向显示
        )
        
        # 生成词云（带异常捕获）
        try:
            wordcloud = wc.generate(text)
        except Exception as e:
            raise RuntimeError(f"词云生成失败: {str(e)}")

        # 显示词云
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        
        # 显式关闭绘图以释放内存
        st.pyplot(fig, clear_figure=True)
        plt.close('all')
        
        # 显示调试信息
        st.caption(f"使用字体路径: {font_path or '系统默认'}")

    except Exception as e:
        st.error(f"词云生成失败: {str(e)}")
        st.markdown("""
            **紧急解决方案：**
            1. [下载微软雅黑字体](https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf)
            2. 将文件重命名为 `SimHei.ttf`
            3. 上传到项目根目录
            """)

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

        # 3. 生成词云（使用改进后的函数）
        st.subheader("☁ 词云图")
        generate_wordcloud(" ".join(words))  # 关键修改点

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