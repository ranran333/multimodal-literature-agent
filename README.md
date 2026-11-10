# Literature QA Agent

面向研究论文的本地多模态问答原型。将 PDF 拆成带来源与页码的文本块，可选为页面生成图表描述并建立多语言向量；通过 LangGraph 检索证据、判断是否读图、调用本地 Qwen2.5-VL，并返回引用和工具轨迹。

> 当前是研究原型。页面引用代表 PDF 阅读器中的**物理页序号**（从 1 开始），不一定等于论文印刷页码。模型可能读错图表或错误理解证据，重要结论请回看原文。

## 流程

```text
PDF → 分页与分块 ──────────────→ BM25 / 可选多语言向量
     └→ 可选：Qwen 页面视觉描述 ─→ 混合检索
问题 → 检索 → LangGraph 路由 → 可选：Qwen 阅读页面图片 → 回答 + [E1] 引用 + 工具轨迹
```

- **解析与溯源**：`pypdf` 提取文字；`pypdfium2` 将需要的页面渲染成图片。每个证据保留 PDF 名称、文档标识与物理页码，并校验源文件是否变化。
- **检索**：中英文词项 BM25；可选使用 `paraphrase-multilingual-MiniLM-L12-v2` 建立向量，在页面级别通过 reciprocal rank fusion 融合。中文问题在英文语料中缺少可靠词项匹配时优先采用向量检索；视觉描述与原文提取结果区分。
- **Agent**：LangGraph 在文本与视觉路径间路由，记录检索页面、路由来源、读图和耗时；模型路由输出无法解析时使用关键词回退。
- **回答与评测**：要求回答使用有效证据标签；无法给出有效标签时返回提示。提供页级 Hit@K、答案字符串匹配、路由和引用页面检查。标签有效不等于引文内容一定支持答案。

默认网页使用自建的解析与检索流程。项目另有一个独立的 **RAG-Anything 实验入口**，已用合成论文验证文字和图表问答，真实论文对照仍在进行；尚未实现 LoRA/SFT。代码 MIT 授权，模型和依赖遵循各自许可证。

## 安装

Python 3.11 推荐。以下示例针对 Linux / NVIDIA GPU；PyTorch 安装方式请根据本机 CUDA 驱动参考[官方选择器](https://pytorch.org/get-started/locally/)。

```bash
git clone https://github.com/ranran333/multimodal-literature-agent.git
cd multimodal-literature-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo,dev]'

# CUDA 12.8 示例；按实际环境选择 CPU 或其他 CUDA 版本
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[model]'
```

模型默认 `Qwen/Qwen2.5-VL-3B-Instruct`，首次使用会从 Hugging Face 下载权重；可通过 `--model Qwen/Qwen2.5-VL-7B-Instruct` 切换到 7B，具体显存取决于输入图像大小和推理配置。默认缓存位于项目的 `.cache/huggingface`，可通过 `HF_HOME` 覆盖。在共享机器上先用 `nvidia-smi` 检查负载，并用 `CUDA_VISIBLE_DEVICES` 选择空闲卡。

如果已用 Conda 在项目内创建环境，先激活该环境，再执行安装命令即可。

## 网页界面

网页可上传 PDF、选择单篇或全部文献提问，并在点击引用后预览对应原文页。每个问题独立检索；页面中显示的是 PDF 物理页码。

在 Linux 或 macOS 上启动服务。以下以项目内已有模型为例；先用 `nvidia-smi` 查看空闲显卡，按实际情况替换编号：

```bash
CUDA_VISIBLE_DEVICES=0 paper-agent web \
  --model models/Qwen2.5-VL-3B-Instruct \
  --embedding-model models/multilingual-MiniLM
```

服务默认创建仅当前系统用户可访问的 `data/web/agent.sock`。如果服务在远程服务器上，在**本机另一个终端**建立 SSH 隧道，将下面的路径和主机名换成实际值：

```bash
ssh -N -L 127.0.0.1:8765:/absolute/path/to/project/data/web/agent.sock user@server
```

保持隧道终端运行，再用本机浏览器访问 `http://127.0.0.1:8765`。服务和隧道分别可以用 `Ctrl+C` 停止。套接字路径过长时，可以通过 `paper-agent web --socket /较短路径/agent.sock` 指定其他路径，并在隧道命令中使用相同路径。

单份 PDF 上限为 50 MB、100 页。默认建立文本和多语言向量索引；上传时可选择“增强图表检索”，逐页生成视觉描述，这需要加载视觉模型并可能耗费较长时间。未开启时，图表问题仍可读取检索到的页面图片，但页面定位主要依赖现有文本。模型首次提问时加载并在进程内复用。

上传文件和网页索引保存在 `data/web/`，不提交到 Git。使用已有本地模型时，问答不调用远程模型 API。当前网页是单用户研究原型，不提供用户管理；不要将 SSH 转发绑定到公共网络。

## 最小演示

演示文档由脚本生成，数据和图表完全虚构：

```bash
python scripts/create_demo.py
paper-agent ingest data/raw --index data/processed/index.json
paper-agent search 'How many research documents are in the benchmark?' --top-k 2
paper-agent evaluate examples/synthetic-eval.jsonl --top-k 2
```

问答需要安装模型依赖和下载模型：

```bash
CUDA_VISIBLE_DEVICES=0 paper-agent ask 'In Figure 1, which bar is taller, Aspen or Cedar?' --mode vision
```

输出 JSON 包含 `answer`、`citations` 和 `trace`。`--mode auto` 让模型选择路径，`--mode text` 固定文本基线，`--mode vision` 强制读取检索到的页面；`--output outputs/result.json` 可保存结果。

连续提问可以使用 `CUDA_VISIBLE_DEVICES=0 paper-agent chat`，模型只需加载一次；每个问题独立检索和回答，输入 `exit` 退出。

对真实文献建立图文索引：

```bash
# 将有权使用的 PDF 放到 data/raw/。图表描述按页运行，文档越多耗时越长。
CUDA_VISIBLE_DEVICES=0 paper-agent ingest data/raw --describe-pages --dense
paper-agent search 'figure 2 results'
CUDA_VISIBLE_DEVICES=0 paper-agent ask 'What trend does Figure 2 show?' --mode auto
```

`--dense` 首次下载多语言编码模型，并在 CPU 上计算向量。建立视觉描述时不要更改正在运行的其他人的 GPU 任务。重建索引会完整处理输入 PDF；原始 PDF、索引、输出、权重和缓存默认不提交 Git。

想逐步理解代码并自己动手验证？参见[中文实践指南](docs/LEARNING_GUIDE_zh.md)。

## RAG-Anything 实验

实验使用独立 Python 环境和存储目录：通过 MinerU 解析论文，在 RAG-Anything / LightRAG 中建立索引，再将检索到、能核实页码的内容交给现有 Agent。图表按图片处理，文字按页分块；缺少可靠页码或跨页的块不会被当作页面引用。当前实验入口在命令行，网页仍运行原有流程。

隔离环境安装、运行命令、页码处理和已验证范围见 [RAG-Anything 实验说明](docs/RAG_ANYTHING_EXPERIMENT_zh.md)。[接入计划](docs/RAG_ANYTHING_PLAN_zh.md)和[当前方案的探索性基线](docs/RAG_ANYTHING_BASELINE_zh.md)记录了具体步骤、评测题目及结果。真实论文对照尚未完成，不能据此声称 RAG-Anything 有性能提升。

## 评测

`examples/synthetic-eval.jsonl` 仅有 3 个**合成示例问题**，用于验证流程，不是对真实文献性能的估计。`examples/rag-paper-eval.jsonl` 是基于 [RAG 原论文](https://arxiv.org/abs/2005.11401)人工编写的 6 道开发问题，主要用于检查检索与回答链路；它们参与了实现调试，不能视为独立测试集。每行格式：

```json
{"question":"问题", "document":"paper.pdf", "page":2, "answer_contains":["关键词"], "route":"vision"}
```

`document` 与 `page` 必填，用于检索的页级 Hit@K；`answer_contains` 和 `route` 可选。加 `--qa` 将生成答案，并报告字符串匹配率、路由准确率及是否包含标注页的引用。它们是自动化代理指标：**字符串匹配和引用页命中不能证明语义正确或引用确实支持答案**。建议在独立真实论文集上人工逐条标注和审核后报告实际指标。

自动化单元测试使用合成 PDF 和模拟模型，主要验证流程、来源和引用检查，并不评估真实模型的回答质量。

如需运行真实论文的开发样例，先将上述 RAG 原论文 PDF 保存为 `data/raw/rag-original.pdf`，再执行：

```bash
paper-agent ingest data/raw/rag-original.pdf --index data/processed/rag-index.json
paper-agent evaluate examples/rag-paper-eval.jsonl --index data/processed/rag-index.json --top-k 3
```

```bash
paper-agent evaluate examples/synthetic-eval.jsonl --top-k 2 --qa --output outputs/eval.json
python -m pytest -q
```

## 局限

- 扫描件需要 OCR。双栏阅读顺序、表格结构、公式与密集小字可能提取不完整。
- 页面视觉描述只覆盖已索引页面，生成描述也可能出错；索引与图片并非独立真值。
- 空的检索结果会拒答；有效的引用标签只验证格式及页码映射，仍需要人工判断证据支持性。
- 索引记录本地 PDF 绝对路径。移动 PDF 后请重建索引；请勿公开原文、索引或个人数据。
- 本项目未进行模型微调、未声明真实数据集上的性能提升。

## License

MIT. See [LICENSE](LICENSE). Third-party libraries and model weights have their own terms.
