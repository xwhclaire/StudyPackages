# Assignment 5 LMDeploly大模型量化部署实验
## 基础作业
- 使用LMDeploy以本地对话部署InternLm-Chat-7B生成的300字小故事

  ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/a68b2d3f-e3d8-44f6-95e6-a9bccbb86dfa)

- 以网页Gradio生成的
  - 没有API Server，TurboMind直接与Gradio通信（TurboMind推理作为后端）

  ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/c6ed3bc0-c81f-47d3-9f58-2e61b7e3858c)

  - Gradio + API Server，Gradio为Client（TurboMind服务作为后端）
  终端界面
  
  ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/801a5f02-1e38-4730-83cd-f8e8d23f2bfb)


  故事生成

  ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/f78cb444-0319-4b30-9fac-dac4e7e30181)


## 进阶作业 
pending...
- 将第四节课训练自我认知小助手模型使用 LMDeploy 量化部署到 OpenXLab 平台。
- 对internlm-chat-7b模型进行量化，并同时使用KV Cache量化，使用量化后的模型完成API服务的部署，分别对比模型量化前后（将 bs设置为 1 和 max len 设置为512）和 KV Cache 量化前后（将 bs设置为 8 和 max len 设置为2048）的显存大小。  
- 在自己的任务数据集上任取若干条进行Benchmark测试，测试方向包括：  
（1）TurboMind推理+Python代码集成  
（2）在（1）的基础上采用W4A16量化  
（3）在（1）的基础上开启KV Cache量化  
（4）在（2）的基础上开启KV Cache量化  
（5）使用Huggingface推理
