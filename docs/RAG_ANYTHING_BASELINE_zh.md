# RAG-Anything 接入前：当前系统的探索性基线

记录日期：2026-11-10  
状态：当前系统已完成一次检索与问答测试；RAG-Anything 对照尚未完成。

## 本次测了什么

从 [ViT](https://arxiv.org/abs/2010.11929) 和 [CLIP](https://arxiv.org/abs/2103.00020) 两篇公开论文中整理了 19 道题：正文 6 道、图表 8 道、表格 5 道。每题都记录了参考答案和至少一个支持答案的 PDF 物理页码；题目与来源清单在 `examples/raganything_eval/`。

这是为本项目整理的小样本，标注和本次复核由同一执行过程完成，尚未经独立人工复核。首轮测试后，为一道题补充了另一个同样支持答案的页面，并将评测口径改为先选择对应论文；初版结果和修订后结果均保留在服务器。它不应被当作独立盲测或总体准确率。

## 配置

- **问题范围**：每道题先选择其对应的单篇论文，模拟网页中选择一篇文献后提问。
- **索引与检索**：现有 PDF 分页及文本分块、BM25 加多语言 MiniLM 向量、页面级融合；返回最多 4 个不同页面。
- **视觉索引**：关闭逐页图表描述，和网页上传时未勾选“增强图表检索”的默认设置一致。
- **问答**：Qwen2.5-VL-3B-Instruct，自动选择文字或视觉路径，视觉路径最多读取前 2 个候选页面。
- **环境**：服务器上的 `wjr3`；本次问答使用一张当时空闲的 NVIDIA L20。

## 结果

| 题目类型 | 题数 | 前 1 页命中标注页 | 前 4 页命中标注页 | 答案经本次逐条核对正确 | 答案正确且引用页支持 |
|---|---:|---:|---:|---:|---:|
| 正文 | 6 | 1 | 5 | 5 | 4 |
| 图表 | 8 | 0 | 2 | 1 | 1 |
| 表格 | 5 | 4 | 5 | 3 | 3 |
| 合计 | 19 | 5 | 12 | 9 | 8 |

页级命中表示候选页面中包含至少一个已标注的支持页，不代表回答已经正确。逐条答案与引用核对由本次执行者对照原文完成，尚需独立复核；没有回答或回答错误计为未通过。逐题判断与原因记录在 `examples/raganything_eval/baseline_review.jsonl`。标注页可能不穷尽所有可以支持答案的页面。

程序的字符串匹配得到 8/19，不能直接作为答案正确数：例如 ViT 算力比较题的答案使用了 Unicode 减号和空格，意思正确却未匹配预设字符串。本报告的 9/19 是逐条核对结果，不是该自动指标。

## 发现的几个问题

1. **图表页经常没有进入候选范围。** 本次 8 道图表题中，只有 2 道的标注页出现在前 4 页。没有逐页图表描述时，检索主要依赖可提取文字和图注。
2. **答对和引用对是两件事。** CLIP 的 WIT 缩写题回答正确，但引用的是第 45 页；该页没有给出缩写展开，不能用作这条答案的依据。
3. **检索命中后仍可能读错。** ViT 的训练 batch size 题在候选中找到了支持页，回答却给出了微调时使用的 512，而不是题目所问的 4096。
4. **自动字符串匹配会漏判。** 数值、连字符、空格或同义表述不同，都可能让正确回答被记为未匹配。

## 复现路径

服务器项目目录下的实验目录：`data/raganything/eval/`。

- `pdfs/`：本次使用的论文 PDF，不提交到 Git。
- `baseline-index.json`：未开启图表描述的现有系统索引。
- `questions.jsonl`：首轮标注；`questions-v2.jsonl`：补充有效页码后的版本。
- `baseline-retrieval.json` / `baseline-retrieval-v2.json`：首轮全库检索结果及修订标注后的全库结果，仅作开发记录；部分问题未注明论文名，存在歧义。
- `baseline-retrieval-scoped.json`：单篇范围检索结果。
- `baseline-qa-scoped.json`：单篇范围下的真实模型回答、引用与轨迹。该文件保留在服务器本地。

核心命令如下，先在 `wjr3` 中激活已有 `.conda`，并按实时显卡情况设置 `CUDA_VISIBLE_DEVICES`：

```bash
paper-agent ingest data/raganything/eval/pdfs \
  --index data/raganything/eval/baseline-index.json \
  --dense --embedding-model models/multilingual-MiniLM

paper-agent evaluate data/raganything/eval/questions-v2.jsonl \
  --index data/raganything/eval/baseline-index.json \
  --top-k 4 --scope-document --qa \
  --model models/Qwen2.5-VL-3B-Instruct \
  --output data/raganything/eval/baseline-qa-scoped.json
```

模型回答会有波动，耗时及显存也受输入长度和共享资源状态影响。RAG-Anything 应使用同一批论文、问题和单篇范围进行对照，再报告其真实结果。
