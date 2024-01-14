# Lecture 3 基于InternLM和LangChain搭建个人知识库
## 大模型开发范式
### LLM的局限性
1. 知识实效性受限：获取最新知识的问题有待解决
2. 专业能力有限：垂直领域大模型的打造有需求但仍待开发
3. 定制化成本高：打造个人专属LLM
### RAG VS Finetune
| RAG | Finetune |
| ----------- | ---------- |
| 低成本 | 可个性化微调 |
| 可实时更新 | 知识面覆盖广 |
| 受基座模型影响大 | 成本高昂 |
| 单次回答知识有限 | 无法实时更新 |
### RAG 检索增强生成
pic1

## LangChain简介
为LLM提供通用接口而简化程序开发的开源工具
### 基于LangChain搭建RAG应用
pic2

## 构建向量数据库
### 基本步骤
- 加载源文件
  - 确定源文件类型，带格式文本转化为无格式字符串
- 文档分块
  - 按字符串长度分割
  - 也可手动控制分割块长度和重叠区长度
- 文档向量化
  - Embedding模型来向量化
  - 使用向量数据库，轻量级的Chroma

## 搭建知识库助手
- 将InternLM接入LangChain
- 构建检索问答链
  - LangChain提供检索问答链模版，可自动实现知识检索、Prompt嵌入、LLM问答全部流程
  - 检索问答链上游：接入InternLM的自定义LLM、已构建的向量数据库
  - 调用检索问答链
  pic3
- RAG方案优化建议
  - 性能核心受限因素：检索精度、Prompt性能
  - 优化策略：
    - 检索角度：
      - 基于语义而非字符长度进行分割，保证每一个chunk的语义完整
      - 给每一个chunk生成概括性索引，检索时匹配索引
  - Prompt角度：
    - 迭代优化Prompt策略

## Web Demo部署
使用Gradio、Streamlit等支持简易Web部署的框架

## 实战部分
### 环境配置
- InternLM模型部署及下载
- LangChain环境配置
  - 使用开源词向量模型Sentence Transformer，通过hugging face镜像工具下载
- NLTK相关资源下载
- 同步本项目代码
### 知识库搭建
整体代码由三部分构成：数据收集、数据加载、构建向量数据库
- 数据收集，找出所有后缀名为.md或.txt的文件。定义函数`get_files`
```
{
import os 
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 后缀为.md将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                # 后缀为.txt同样加入结果列表
                file_list.append(os.path.join(filepath, filename))
    return file_list
}
```
- 数据加载，使用FileLoader对象加载目标文件。针对性调用FileLoader以应对不同类型的文件
```
{
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader

def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs
}
```
- 向量数据库构建：先对文本分块，再对文本进行向量化
  - 分块大小500，块重叠长度150
  ```
  {
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
  }
  ```
  - Sentence Transformer 进行文本向量化
  ```
  {
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
  }
  ```
  - 选择Chroma为向量数据库 
  ```
  {
    from langchain.vectorstores import Chroma
    
    # 定义持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
        )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()
  }
  ```

### InternLM接入LangChain
基于本地部署的InternLM并继承LangChain的LLM类，定义一个InternLM的LLM子类。
从LangChain.llms.base.LLM类继承一个子类，重写`__init__`和`_call`函数

### 构建检索问答链
- 加载向量数据库：直接通过Chroma及词向量模型来加载
  ```
  {
   from langchain.vectorstores import Chroma
   from langchain.embeddings.huggingface import HuggingFaceEmbeddings
   import os
   
   # 定义 Embeddings
   embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
   
   # 向量数据库持久化路径
   persist_directory = 'data_base/vector_db/chroma'
   
   # 加载数据库
   vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
    )
  }
  ```
- 实例化自定义LLM与Prompt Template
  ```
  {
    from langchain.prompts import PromptTemplate
    
    # 我们所构造的 Prompt 模板
    template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {context}
    ···
    如果给定的上下文无法让你做出回答，请回答你不知道。
    有用的回答:"""
    
    # 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
  }
  ```
- 调用LangChain提供的检索问答链构造函数，基于自定义的LLM、Prompt Template和向量知识库来构建基于InternLM的检索问答链
  ```
  {
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
  }
  ```