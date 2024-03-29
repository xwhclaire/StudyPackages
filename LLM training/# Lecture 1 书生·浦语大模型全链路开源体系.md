# Lecture 1 书生·浦语大模型全链路开源体系
## 1. 从模型到应用
![1-1](https://github.com/xwhclaire/StudyPackages/assets/34467524/7e17801d-662b-4bfd-8a00-c9236c610c88)
## 2. 全链条开源开放体系

### 1）数据 
- 书生·万卷 1.0
多模态融合、精细化处理、价值观对齐
- OpenDataLab

### 2）预训练
- 高可扩展：支持从8卡到千卡训练 加速效率92%
- 极致性能优化：加速50%
- 兼容主流：无缝接入HuggingFace等技术生态，支持各类轻量化技术
- 开箱即用：支持多种规格语言模型，修改配置即可训练
### 3）微调
- 增量续训
- 有监督微调
  - 全量参数微调
  - 部分参数微调
- 高效微调框架 XTuner
![1-2](https://github.com/xwhclaire/StudyPackages/assets/34467524/071278d1-ff2b-4aa9-be92-f9a861a50271)


### 4）评测
- 国内外评测体系
![1-3](https://github.com/xwhclaire/StudyPackages/assets/34467524/54696dca-8828-4806-b36a-07cf48c42dab)

- OpenCompass
  - 6大维度、80+评测集、40w+评测题目
  - 平台架构：4层
    - 工具层：分布式评测、提示词工程、评测数据库上报、评测榜单发布、评测报告生成
    - 方法层：自动化客观评测、基于模型辅助的主观评测、基于人类反馈的主观评测
    - 能力层：通用能力、特色能力
    - 模型层：基座模型、对话模型
  - highlight：丰富模型支持、分布式高效评测、便捷的数据集接口、敏捷的能力迭代

### 5）部署
- LLM特点：内存开销巨大、动态Shape、模型结构相对简单
- 技术挑战：设备、推理、服务
- 部署方案：模型并行、地比特量化、Attention优化、计算和访存优化、Continuous Batching
- LMDeploy
  - 不同接口：python、gPRC、RESTful
  - 轻量化：4bit权重、8bit k/v
  - 推理引擎：turbomind、pytorch
  - 服务：openai-server、gradio、trition inference server

### 6）智能体
- LLM局限性
![1-4](https://github.com/xwhclaire/StudyPackages/assets/34467524/9b3b5fca-cc89-4c96-a075-74ef8d23b9ff)

- 轻量级智能体框架Lagent
  - 支持多种类型的智能体能力
  - 灵活支持多种LLM
  - 简单易拓展，支持丰富的工具
