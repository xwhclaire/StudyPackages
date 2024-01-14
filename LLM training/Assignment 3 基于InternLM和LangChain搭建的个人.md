# Assignment 3 基于InternLM和LangChain搭建的个人知识库
## 基础作业
### 环境配置
- LangChain相关环境配置
pic4 pic5
- NLTK相关资源
pic6  
- 下载代码
pic7

### 知识库搭建
运行脚本`create_db.py`
pic8

### InternLM接入LangChain
将构建LLM子类的代码封装为`LLM.py`，便于在构建检索问答链的过程中从该文件引入LLM子类

### 构建检索问答链并部署Web Demo
运行脚本`run_gradio.py`
pic9
浏览器中打开页面`http://127.0.0.1:7860/`
pic10

## 进阶作业
