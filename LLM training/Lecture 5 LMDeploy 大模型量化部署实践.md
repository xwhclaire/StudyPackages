# Lecture 5 LMDeploy 大模型量化部署实践
## 大模型部署背景
- 特点

![5-1](https://github.com/xwhclaire/StudyPackages/assets/34467524/bea42887-9aa8-41e4-ad43-0017d35e3902)

- 部署挑战和部署方案

![5-2](https://github.com/xwhclaire/StudyPackages/assets/34467524/bd1b7091-6f16-4c6d-93d0-016fdce24ee7)

## LMDeploy简介
LLDeploy是LLM在英伟达设备上部署的全流程解决方案。包括模型轻量化、推理和服务。上层是Python，底层结构通过C++搭建。[项目地址](http://github.com/InternLM/Imdeploy)

![5-3](https://github.com/xwhclaire/StudyPackages/assets/34467524/088f1d2d-b1e3-4b65-bad9-fc6ca5804f78)

### 推理性能：静态推理性能 VS 动态推理性能

![5-4](https://github.com/xwhclaire/StudyPackages/assets/34467524/404b0532-e66e-41e8-bf05-4f538e1dd143)

### 核心功能
- 量化
  降低显存，同样设备下可以容纳更多的并发及更大的长度
  量化效果对比明显：

![5-5](https://github.com/xwhclaire/StudyPackages/assets/34467524/c37b1ab0-ab4f-4d6c-892e-e5f9c1c2eb10)

  - 为什么Weight Only的量化？<br>
      降低显存占用、提升推理速度（大幅降低访存成本，提高Decoding的速度） 
    1. 计算密集（compute-bound）：推理的绝大部分时间消耗在数值计算上；针对计算密集场景，可以通过使用更快的硬件计算单元来提升计算速度，比如量化为W8A8使用INT8 Tensor Core来加速计算。
    2. 访存密集（memory-bound）：推理时，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般是通过提高计算坊存比来提升性能。例如LLM，Decoder Only架构，推理时大部分时间消耗在逐Token生成阶段（Decoding阶段），典型的访存密集型场景。

![5-6](https://github.com/xwhclaire/StudyPackages/assets/34467524/1dc65780-dcb5-4cbd-bbab-e86476e56296)

  -  如何做Weight Only量化？<br>
      AWQ算法，量化为4bit模型。（相较于GPTQ算法，AWQ的推理速度更快，量化的时间更短）
      推理时，先把4bit权重，反量化回FP16（在Kernel内部进行，从Global Memory读取时仍是4bit），依旧使用FP16计算

![5-7](https://github.com/xwhclaire/StudyPackages/assets/34467524/da243ac7-51ab-4e60-95eb-3a2bdcc6efdb)


- 推理引擎TurboMind

![5-8](https://github.com/xwhclaire/StudyPackages/assets/34467524/540b863b-c33f-47fe-87f9-a2bf0f1542d4)


  1. 持续批处理：<br>
    推理请求首先加入到请求队列中；
    若batch中有空闲槽位，从队列拉去请求，尽量填满空闲槽位。若无，继续对当前batch中的请求进行forward；
    batch每forward完一次，判断是否有request推理结束，结束的request，发送结果，释放槽位；
    继续拉取请求填满空位
  2. 有状态的推理<br>
    无状态 vs 有状态

  ![5-9](https://github.com/xwhclaire/StudyPackages/assets/34467524/95b4e912-b771-44d4-8f73-cf64f02031ca)

       历史记录在推理侧缓存的过程：
  
  ![5-10](https://github.com/xwhclaire/StudyPackages/assets/34467524/e004b95b-2aab-4091-ba7a-682bc7922763)

  3. Blocked k/v cache<br>
    状态迁移：

   ![5-11](https://github.com/xwhclaire/StudyPackages/assets/34467524/e480dbd9-2d02-46b8-b85c-a7599170ecdc)

  Free：未被任何序列占用<br>
  Active：被正在推理的序列占用<br>
  Cache：被缓存中的序列占用<br>
    过程展示：

  ![5-12](https://github.com/xwhclaire/StudyPackages/assets/34467524/c52e9be5-0177-4503-be85-b127a313b7d4)

  4. 高性能 cuda kernel
    四个方面优化：
    - Falsh attention 2
     - Fast w4a16，kv8
     - Split-k decoding
     - 算子融合
- 推理服务 api server
   使用`Imdeploy serve api_server InternLM/internlm-chat-7b --model-name internlm-chat-7b --server-port 8080`加载打开api server

   Swagger地址：[http://0.0.0.0.8080](http://0.0.0.0.8080)

## 动手实践环节
### 安装
- 安装lmdeploy，完成环境部署

![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/a386656f-8b18-449d-a6f6-8249f3c68cfe)

### 部署

![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/8f087767-9297-4d26-a30c-39216a226217)

模型推理/服务：主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
Client：可以理解为前端，与用户交互的地方。
API Server：一般作为前端的后端，提供与产品和服务相关的数据和功能支持。
以上的划分是一个相对完整的模型，但在实际中这并不是绝对的。比如可以把“模型推理”和“API Server”合并，有的甚至是三个流程打包在一起提供服务。
- 模型转换
  - 在线转换
    - 在huggingface.co上面通过Imdeploy量化的模型，如 [llama2-70b-4bit](https://huggingface.co/lmdeploy/llama2-chat-70b-4bit), [internlm-chat-20b-4bit](https://huggingface.co/internlm/internlm-chat-20b-4bit)
    - huggingface.co 上面其他 LM 模型，如 Qwen/Qwen-7B-Chat
    - 可以直接启动本地的 Huggingface 模型
  - 离线转换
    - Tensor并行
 
    ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/d1141052-a6ea-4a71-ac22-8c595de26cf4)

    <p align="center">列并行<p>
 
    ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/01585278-f8b5-4b33-8b58-4cd03177fd62)

    <p align="center">行并行<p>
    把一个大的张量（参数）分到多张卡上，分别计算各部分的结果，然后再同步汇总。
- TurboMind 推理+命令本地对话
   先尝试本地对话（Bash Local Chat），下面用（Local Chat 表示）在这里其实是跳过 API Server 直接调用 TurboMind。简单来说，就是命令行代码直接执行 TurboMind。

   ![image](https://github.com/xwhclaire/StudyPackages/assets/34467524/ebb41499-76fb-4be9-962a-323fa9b97456)

- TurboMind推理 + API服务
  ”模型推理/服务“目前提供了 Turbomind 和 TritonServer 两种服务化方式。此时，Server 是 TurboMind 或 TritonServer，API Server 可以提供对外的 API 服务。
- Gradio网页Demo演示
  - TurboMind服务作为后端
  ```bash
  # Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client 
  lmdeploy serve gradio http://0.0.0.0:23333 \
  --server_name 0.0.0.0 \
  --server_port 6006 \
  --restful_api True
  ```
  - TurboMind推理作为后端
  Gradio 也可以直接和 TurboMind 连接
  `lmdeploy serve gradio ./workspace`
- TurboMind推理 + Python代码集成
  lmdeploy 还支持 Python 直接与 TurboMind 进行交互

```python
  from lmdeploy import turbomind as tm
  # load model
  model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
  tm_model = tm.TurboMind.from_pretrained(model_path， model_name='internlm-chat-20b')
  generator = tm_model.create_instance()
  
  # process query
  query = "你好啊兄嘚"
  prompt = tm_model.model.get_prompt(query)
  input_ids = tm_model.tokenizer.encode(prompt)
  
  # inference
  for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]
    
    response = tm_model.tokenizer.decode(res.tolist())
    print(response)
```
