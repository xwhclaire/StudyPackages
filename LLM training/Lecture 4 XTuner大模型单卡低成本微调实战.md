# Lecture 4 XTuner大模型单卡低成本微调实战
## Finetune简介
- 两种LLM下游应用中的微调模式
| 增量预训练 | 指令跟随 |
| ----------- | ----------- |
| 让基座模型学习到一些新知识（垂类领域） | 让模型学会对话模版（人类指令对话） |
| 文章、书籍、代码等 | 高质量对话、问答数据 |
![4-1](https://github.com/xwhclaire/StudyPackages/assets/34467524/28cfa0d3-2c1a-4434-a599-a870be25e579)

  - 指令跟随微调模式
  ![4-2](https://github.com/xwhclaire/StudyPackages/assets/34467524/bc5f3c1b-cd21-41c9-8fcd-ead5c6ba0577)

  ![4-3](https://github.com/xwhclaire/StudyPackages/assets/34467524/fa66d83a-839b-4764-9c25-e2e7b3b9db98)

  不同模型会有不同模板 LIaMa2 VS InternLM
![4-4](https://github.com/xwhclaire/StudyPackages/assets/34467524/d0b35257-b99a-49ac-aff1-3bb78e747dea)

  - 增量预训练微调模式
  对数据添加起始符（BOS）和结束符（EOS）
  ![4-5](https://github.com/xwhclaire/StudyPackages/assets/34467524/83ef64e2-6c12-475c-8d62-7a3764f23286)


- LoRA & QLoRA
  - LoRA（Low-Rank Adaptation）
  在原本Linear旁新增一个支路，包含两个连续的小Linear，即Adapter。能大幅降低训练的显存损耗

    ![4-6](https://github.com/xwhclaire/StudyPackages/assets/34467524/99f266a9-7fa7-4fb7-9fa5-f38e4f2e88e8)

  - 三者对比
    ![4-7](https://github.com/xwhclaire/StudyPackages/assets/34467524/62be518b-f76b-4127-9cc3-aa81062fb3a9)

## Xtuner介绍
![4-8](https://github.com/xwhclaire/StudyPackages/assets/34467524/23ea8ec3-608f-4e4f-8816-e1aa55356fdf)

- 工具类模型对话能力
- 数据处理引擎
  - 多种热门数据集的映射函数
  - 多种对话模版映射函数
专注于数据内容，不必花费精力处理复杂的数据格
  - 多数据样本拼接：增强并行性，充分利用GPU资源
  - 数据集建议使用`json`或者`jsonl`格式 

## 8GB显卡玩转LLM
- Xtuner的两个优化技巧
  - Flash Attention
    ![4-9](https://github.com/xwhclaire/StudyPackages/assets/34467524/dc3867d4-bc99-49bc-b4dc-d0d39da64bc9)

  将Attention计算并行化，避免计算过程中Attention Score NxN的显存占用
  - DeepSpeed ZeRO
    ![4-10](https://github.com/xwhclaire/StudyPackages/assets/34467524/f1bb00fd-fe6c-47f6-9f60-04f1a56e47a6)

  将训练过程中的参数、梯度和优化器状态切片保存，能够在多GPU训练时显著节省显存
  使用时增加`--deepspeed deepspeed_zero3`启动参数
  对于QLoRA算法，则使用`--deepspeed deepspeed_zero2`作为启动参数
- 优化前后显存占用情况对比
    ![4-11](https://github.com/xwhclaire/StudyPackages/assets/34467524/21129145-3119-440a-ab6f-9a0f8584d6bf)

## 实战
### 环境部署
- 安装 xtuner0.1.9 并激活 conda 环境
- 拉取版本源码并从源码安装 XTuner
### 微调工作
- 准备配置文件 `xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .`
  | 模型名 | internlm_chat_7b |
  |-------------|----------------|
  | 使用算法 | qlora |
  | 数据集 | oasst1 |
  | 把数据集跑几次 | 跑3次:e3(epoch3) |
- 模型下载，可以在教学平台直接复制模型，使用`pip install modelscope`拉取 modelscope 库，并从库里下载模型文件`git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3`
- 数据集下载 `cp -r /root/share/temp/datasets/openassistant-guanaco .`
  - 训练前的路径

    ![4-12](https://github.com/xwhclaire/StudyPackages/assets/34467524/e98d49cb-fd64-4ad1-93f7-5f59e33a352b)

- 修改配置文件 `internlm_chat_7b_qlora_oasst1_e3_copy.py` 将模型和训练数据均改为本地路径
- 开始微调，开启deepspeed加速 `xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2` 
  - 训练后的路径
    
    ![4-13](https://github.com/xwhclaire/StudyPackages/assets/34467524/97552d71-3ffd-4519-93a2-ddecde9565c7)
  - 生成 Adapter 文件


### 补充内容
- `tmux`命令中断ssh连接，不中断微调工作
    `tmux new -s finetuned` 在后台进行微调工作
    `tmux attach -t finetune` 重新回到后台进行操作
  
- 
