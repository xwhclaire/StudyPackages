# Assignment 3 基于InternLM和LangChain搭建的个人知识库
## 基础作业
### 环境配置
- LangChain相关环境配置
![3-4](https://github.com/xwhclaire/StudyPackages/assets/34467524/d6f18019-4592-4afb-b113-22c87d1c7726)

![3-5](https://github.com/xwhclaire/StudyPackages/assets/34467524/ebce1208-2547-45f3-a687-c2ab73ae84a7)

- NLTK相关资源

![3-6](https://github.com/xwhclaire/StudyPackages/assets/34467524/f342a152-373c-4803-a2ec-56f23a0b30c7)

- 下载代码

![3-7](https://github.com/xwhclaire/StudyPackages/assets/34467524/24e8a93e-53ef-4a77-8ef6-b5132acc17bf)


### 知识库搭建
运行脚本`create_db.py`

![3-8](https://github.com/xwhclaire/StudyPackages/assets/34467524/62ad85ee-77c9-47d5-b193-07a431962cca)


### InternLM接入LangChain
将构建LLM子类的代码封装为`LLM.py`，便于在构建检索问答链的过程中从该文件引入LLM子类

### 构建检索问答链并部署Web Demo
运行脚本`run_gradio.py`

![3-9](https://github.com/xwhclaire/StudyPackages/assets/34467524/2bc72b3d-f573-4b6e-82d6-33811908d664)

浏览器中打开页面`http://127.0.0.1:7860/`

![3-10](https://github.com/xwhclaire/StudyPackages/assets/34467524/9d00e1f7-eb2e-470b-b309-f1bfbc8d1dcf)

## 进阶作业
