# RAG-Anything 实验说明

本实验在原有 Agent 之外增加一条文献处理和检索路径。网页默认仍使用原有流程，下面的实验通过命令行运行；两种方案不会共用索引。

## 做了什么

1. 使用 MinerU 解析 PDF 中的文字、图片、表格和公式。
2. 用 RAG-Anything / LightRAG 建立索引，文本生成与图片理解接到本地 Qwen2.5-VL，向量接到本地多语言 MiniLM。
3. 每篇 PDF 使用独立工作区。文字在页与页之间分开；MinerU 标为 `chart` 的图片转交图像处理器。
4. 检索后从存储的文本块取回文档与页码。页码从解析器的零起始编号转为 PDF 阅读器中的一始编号；缺失页码、跨页或无法核实来源的块不作为引用。
5. 把可定位的页面交给现有 Agent，由它决定是否看图、回答并展示引用。模型写出的图表描述本身也可能出错，需要回看原文。

## 已经验证到哪里

在服务器的独立环境中，使用 RAG-Anything 1.4.4、LightRAG 1.4.16、MinerU 3.4.5 和本地 Qwen2.5-VL-3B-Instruct，处理了一篇两页的**自造论文**。文字问题回答了 120，并引用第 1 页；图表问题回答 Cedar 的柱子更高，并引用第 2 页。这个样例只用于检查链路，不能代表真实论文的效果。

两篇真实论文 ViT 和 CLIP 的处理及对照正在进行。[基线报告](RAG_ANYTHING_BASELINE_zh.md)记录的是接入前原有系统的结果，不能当作 RAG-Anything 的结果。

真实论文建索引时，日志中出现了部分实体关系抽取的输出格式警告。这可能影响图谱的完整性；需要结合最终检索结果检查，不能把成功建索引等同于图谱准确。

## 安装

请为实验新建独立的 Python 3.11 环境，不要直接在正在运行网页的环境里更换依赖。下面假设已经按项目 README 安装好适合本机 CUDA 的 PyTorch，并在新环境中安装了项目的 `model` 依赖：

```bash
python -m pip install -e '.[model]'
python -m pip install -r requirements-raganything.txt
python -m pip check
```

首次使用可能下载 MinerU 解析模型和 LightRAG 使用的分词文件，也需要本地可用的 Qwen 和 MiniLM 权重。下载体积、耗时及 GPU 显存与具体版本、论文和图像大小有关。在共享服务器上先用 `nvidia-smi` 查看负载，再选择空闲 GPU。MinerU 模型来源可用 `MINERU_MODEL_SOURCE=modelscope` 或按实际网络设置；模型和依赖分别遵循其许可证。

## 跑一篇论文

在项目根目录准备有权使用的 PDF。先建立原有格式的文档清单，实验脚本从中读取文件路径、摘要和页数，用来校验来源：

```bash
paper-agent ingest data/raw/paper.pdf --index data/raganything/source-index.json

# 0 是当前可用的显卡编号示例；请按 nvidia-smi 的结果修改。
CUDA_VISIBLE_DEVICES=0 MINERU_MODEL_SOURCE=modelscope \
  MINERU_TOOLS_CONFIG_JSON="$PWD/data/raganything/mineru.json" \
  MODELSCOPE_CACHE="$PWD/data/raganything/modelscope-cache" \
  python scripts/raganything_experiment.py \
  --index data/raganything/source-index.json \
  --model models/Qwen2.5-VL-3B-Instruct \
  --embedding-model models/multilingual-MiniLM \
  ingest --document paper.pdf

CUDA_VISIBLE_DEVICES=0 MINERU_MODEL_SOURCE=modelscope \
  MINERU_TOOLS_CONFIG_JSON="$PWD/data/raganything/mineru.json" \
  MODELSCOPE_CACHE="$PWD/data/raganything/modelscope-cache" \
  python scripts/raganything_experiment.py \
  --index data/raganything/source-index.json \
  --model models/Qwen2.5-VL-3B-Instruct \
  --embedding-model models/multilingual-MiniLM \
  ask --document paper.pdf '这篇论文的主要方法是什么？'
```

模型路径应替换成自己的本地路径。`--document` 必须与文档清单中的文件名一致。PDF 原文、模型、索引和运行日志都不应提交到公开仓库；文件移动或内容变化后需重建清单和实验索引。

## 对照评测

`examples/raganything_eval/` 保存了题目、来源信息和原有系统的逐题复核。对两篇论文分别完成实验索引后，可用相同的 `questions-v2.jsonl`、单篇范围和前 4 个去重页面进行对照。输出会记录每题的检索页、无法定位页码的块、耗时及错误；加 `--qa` 才会运行回答。回答和引用是否真的正确仍需逐题对照原文。详见[接入计划](RAG_ANYTHING_PLAN_zh.md)。
