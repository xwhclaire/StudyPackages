# Lecture 6 OpenCompass 大模型评测
## 关于评测的3个问题
### Why
类型各异\数量日益增长的大模型需要建立统一的模型评测。
- 模型选型
pic1
- 模型能力提升
- 真实应用场景效果评测
pic2
### What
- 知识、推理、语言
- 长文体、智能体、多轮对话
- 情感、认知、价值观
以大语言模型为例：  
pic3
### How
- 自动化客观评测
  - 客观评测
   以找到关键词为标准
pic4
  - 主观评测
  依赖于人类评价的主观评测不现实，于是产生了自动化客观评测
pic5
  - 提示词工程
  提高鲁棒性（稳健性）
- 主流大模型评测框架
pic6

## OpenCompass简介
### OpenCompass能力框架
pic7
### OpenCompass开源评测平台架构
pic8
### 评测流水线设计
pic9
### 实际应用
- 大模型能力对比
pic10
- 垂直领域探索
  - 法律领域
   pic10
  - 医疗领域
   pic11

## 大模型评测领域的挑战
pic12

## 实战部分
- 安装及数据准备
  pic
  <p align="center">支持的数据集和模型<p>
- 启动评测
  - 命令解析
  ```bash
  --datasets ceval_gen \
  --hf-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace 模型路径
  --tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
  --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
  --model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
  --max-seq-len 2048 \  # 模型可以接受的最大序列长度
  --max-out-len 16 \  # 生成的最大 token 数
  --batch-size 4  \  # 批量大小
  --num-gpus 1  # 运行模型所需的 GPU 数量
  --debug
  ```
  - `run.py`的使用
  - `outputs`中的时间戳、predictions、summary
  - 预定义模型配置
    - `config/models`下，通过`--models`指定相关文件，或者在`demo.py`脚本中修改models的list
    - `config/datasets`下，通过`--datasets`明确相关数据集，或者通过继承在配置文件中导入相关配置
- 可视化评估结果
在`outputs/20240123_152952/summary`中的`.txt`文件中可以找到可视化列表