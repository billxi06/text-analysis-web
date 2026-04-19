# 小说文本分析系统 (Text Analysis Web)

# 网站url https://watercomputer.streamlit.app/

基于 Streamlit 构建的小说文本深度分析与可视化 Web 应用。用户只需上传 `.txt` 格式的小说文本，系统即可自动进行中文分词，并一键生成词频统计、专属词云图以及基于 LDA 模型的主题关键词提取。

## 核心功能
1.智能文本解析：支持任意 `.txt` 文件上传，内置安全解码机制，自动识别并处理不同文件编码（UTF-8, GBK 等），拒绝乱码。

2.高频词汇统计：利用 Jieba 精准分词，快速提取并展示小说中出现频率最高的前 20 个核心词汇。

3.可视化词云图：自动生成美观的词云图，内置强大的中文字体查找与回退机制，确保在 Windows/macOS/Linux 下均能完美显示中文。

4.LDA 主题分析：深度挖掘文本内涵，自动提取出 3 个核心主题及对应的关键词权重。

# 技术栈
1.Web 框架：Streamlit

2.自然语言处理：Jieba (分词), Gensim (LDA 主题模型)

3.数据处理与可视化：Pandas, WordCloud, Matplotlib

4.其他工具：Chardet (编码检测)

# 快速开始

1.克隆项目

bash

git clone [https://github.com/bailixx/text-analysis-web.git](https://github.com/bailixx/text-analysis-web.git)

cd text-analysis-web

网站页面图片

<img width="835" height="905" alt="网站页面前端" src="https://github.com/user-attachments/assets/fdda7003-ae60-42a2-8d7f-150a05cf659a" />

分析结果图片

![小说文本分析网站分析结果1](https://github.com/user-attachments/assets/bc663618-3ce0-4849-b733-845282776c42)

![小说文本分析网站分析结果2](https://github.com/user-attachments/assets/4e0234bd-a0e1-4871-98f8-a8bb0e5697d6)

![小说文本分析网站分析结果3](https://github.com/user-attachments/assets/f75b9cf9-6e1e-4039-bb77-3c3a04b011cc)



