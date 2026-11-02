# 动手理解这个项目

这份指南帮助你运行并理解项目。命令在项目根目录执行；若使用 Conda，先激活自己的环境。初次下载模型、处理 PDF 或推理需要一定时间。

## 1. 先看清系统如何工作

1. **索引**：程序读取 PDF 的每一页，按段落将文字切成有重叠的小块，并保存文档名和物理页码。可选地让视觉模型描述每页，还可以为文本计算向量。
2. **检索**：收到问题后，用 BM25 匹配关键词；有向量索引时，再把向量排序与关键词排序融合。程序从不同页面选出候选证据。
3. **路由**：LangGraph 根据问题决定走文字路径还是读图路径。也可以手动指定 `text` 或 `vision`，方便对照。
4. **读图与回答**：视觉路径会把候选页面渲染为图片，交给本地 Qwen2.5-VL 阅读。模型根据检索文本和读图结果回答，并使用 `[E1]` 等标签引用页面。
5. **检查结果**：程序检查标签是否有效，并保存工具轨迹；标签有效并不保证证据真的支持答案，需要回到 PDF 核对。

对应代码分别在 `src/literature_agent/index.py`、`retrieval.py`、`agent.py` 和 `models.py`。

## 2. 先做一个小实验

先按 README 安装依赖，再生成虚构的两页论文，建立一个单独的演示索引：

```bash
python scripts/create_demo.py
paper-agent ingest data/raw/synthetic-paper.pdf --index data/processed/demo-index.json
paper-agent search 'How many research documents are in the benchmark?' --index data/processed/demo-index.json --top-k 2
```

观察检索结果的 `document`、`page` 和 `excerpt`。PDF 物理页码从 1 开始，可能与论文中印刷的页码不同。

运行问答前，先用 `nvidia-smi` 查看 GPU 状态，选择有足够空闲显存的设备。下面的 `1` 只是示例，请按实际情况修改。如果模型已经下载到 `models/Qwen2.5-VL-3B-Instruct`，可直接使用本地路径：

```bash
CUDA_VISIBLE_DEVICES=1 paper-agent ask \
  'In Figure 1, which bar is taller, Aspen or Cedar?' \
  --index data/processed/demo-index.json \
  --model models/Qwen2.5-VL-3B-Instruct --mode vision --top-k 1
```

首次运行会加载模型。打开输出中的 `trace`，检查检索到的是哪一页、是否有 `read_page_image`、视觉模型记录了什么，以及最终引用是否指向图所在页。再打开生成的 PDF，自己核对柱形图。可以把 `--mode vision` 改为 `--mode text` 对照；不同路径的结果可能相同，也可能不同。

连续提问时，使用 `paper-agent chat --index data/processed/demo-index.json --model models/Qwen2.5-VL-3B-Instruct`，模型只加载一次。每个问题独立检索，当前版本不维护多轮对话记忆。

## 3. 换成自己的论文

把有权使用的 PDF 放进 `data/raw/`，然后重新建立索引。例如：

```bash
paper-agent ingest data/raw --index data/processed/papers-index.json
paper-agent search '你的问题' --index data/processed/papers-index.json
```

添加 `--dense` 可以建立多语言向量；添加 `--describe-pages` 可以预先生成页面视觉描述，需同时指定合适的 `--model` 和 GPU。文档越多、页面越复杂，处理时间越长。原文、索引、模型和结果默认不会提交到 Git。

试着提出一个答案在正文中的问题和一个必须看图才能回答的问题，分别记录回答、引用页和原文核对结果。如果检索没有找到正确页面，先改写问题或调整 `--top-k`；如果找到页面但模型回答错误，检查读图记录并对照原文。

## 4. 如何理解评测

```bash
paper-agent evaluate examples/synthetic-eval.jsonl --index data/processed/demo-index.json --top-k 2
```

页级 Hit@K 表示正确页面是否出现在前 K 个检索结果中。示例只有 3 个合成问题，适合检查流程，不能代表真实论文上的性能。要评价实际效果，应准备未参与开发的论文和问题，人工标注答案与证据页，并复核模型回答及引用。不要把示例分数写成真实数据集的效果。
