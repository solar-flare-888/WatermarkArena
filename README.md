# WatermarkArena

大语言模型文本水印 **攻防对抗** 后端 —— 基于 [MarkLLM](../MarkLLM-main) 实现，符合《大语言模型文本水印攻防对抗赛题》接口规范。

- **攻防对抗**：全对阵 (Attack × Defense) 矩阵评测（赛题第十二章）。
- **标准 OpenAI 接口**：所有大模型访问统一走 `openai` SDK，兼容 OpenAI / Azure / vLLM / Ollama 等任意 OpenAI 兼容端点。
- **攻防解耦**：攻击方法与防御算法通过独立注册表 (registry) 注册，互不感知，可任意配对。
- **本地数据集**：`download_datasets.py` 将数据集下载/构建到本地 `data/`，覆盖赛题要求的全部任务类型。
- **复用 MarkLLM**：水印生成/检测算法与文本编辑攻击均封装自 MarkLLM；同时提供纯 API 版本以便无本地模型时也可端到端运行。

---

## 目录结构

```
WatermarkArena/
├── arena/                      # 核心后端
│   ├── env.py                  # 标准 OpenAI 接口的 LLM 访问环境 (LLMEnv)
│   ├── defense_base.py         # 防御接口 (WatermarkDefense / WatermarkDetector)，对齐赛题
│   ├── attack_base.py          # 攻击接口 (attack(env, task))，对齐赛题
│   ├── registry.py             # ★ 攻防解耦的注册表
│   ├── config.py               # config.yaml 加载
│   ├── dataset.py              # 本地数据集加载 (PromptDataset / NaturalTextDataset)
│   ├── download_datasets.py    # 数据集下载/构建到本地
│   ├── evaluation.py           # AUC / TPR@FPR / ASR / 语义相似度 / 资格线
│   ├── confrontation.py        # 全对阵 Attack×Defense 运行器
│   ├── defenses/
│   │   ├── markllm_defense.py  # 封装 MarkLLM 水印算法 (KGW/Unigram/SWEET/EXP/...)
│   │   └── api_native_defense.py # 纯 OpenAI 接口的 Best-of-N 绿名单水印
│   └── attacks/
│       ├── markllm_attack.py   # 封装 MarkLLM 文本编辑攻击
│       └── api_native_attack.py# 纯 OpenAI 接口的 LLM 改写/回译攻击
├── scripts/
│   ├── run_confrontation.py    # 主入口：跑全对阵矩阵
│   ├── eval_single.py          # 单组攻防评测/调试
│   └── list_methods.py         # 列出已注册的攻击/防御
├── submission/                 # 赛题第二十一章提交模板
│   ├── defense_template/       # watermark.py + detector.py + config.yaml
│   └── attack_template/        # attack.py + config.yaml
├── data/                       # 本地数据集 (运行 download 后生成)
├── outputs/                    # 评测报告输出
├── config.yaml                 # 主配置
├── requirements.txt
└── README.md
```

> 本项目**单独存放**于 `WatermarkArena/`，与 `MarkLLM-main/` 平级；运行时会自动把 `MarkLLM-main` 加入 `sys.path` 复用其算法。

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r WatermarkArena/requirements.txt
```

### 2. 配置大模型访问（标准 OpenAI 接口）
编辑 `config.yaml`，或设置环境变量：
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可换成任意兼容端点
export OPENAI_MODEL="gpt-3.5-turbo"
```

要使用 MarkLLM 的 logits 水印算法（KGW/SWEET/...），还需配置本地模型：
```bash
export ARENA_LOCAL_MODEL="Qwen/Qwen2.5-1.5B"
export ARENA_DEVICE="cpu"     # 或 cuda
```
未配置本地模型时，可使用 `BlackBoxGreenList` 防御（纯 API，端到端可跑）。

### 3. 下载数据集到本地
```bash
cd WatermarkArena
python -m arena.download_datasets                 # 含 HF 在线下载
python -m arena.download_datasets --no-hf          # 纯离线（仅本地+内置）
```
生成 `data/prompts.jsonl` 与 `data/natural_texts.jsonl`，覆盖：开放式问答、摘要、改写、知识问答、推理、创意写作、结构化生成、长文本、中英文混合。

### 4. 列出已注册的攻击 / 防御
```bash
python scripts/list_methods.py
```

### 5. 运行攻防对抗
```bash
# 全对阵（所有攻击 × 所有防御）
python scripts/run_confrontation.py --max-samples 10

# 指定子集
python scripts/run_confrontation.py \
    --attacks LLMRewrite,Identity \
    --defenses BlackBoxGreenList \
    --max-samples 10

# 仅评测防御（AUC/质量保持率，不跑攻击）
python scripts/run_confrontation.py --no-matrix --defenses BlackBoxGreenList
```

输出示例：
```
===== ASR Matrix (Attack Success Rate) =====
Attack\Defense   BlackBoxGreenList  KGW   ...
LLMRewrite       0.700              ...
Identity         0.000              ...
```

### 6. 单组调试
```bash
python scripts/eval_single.py --defense BlackBoxGreenList --attack LLMRewrite --max-samples 3
```

---

## 架构与解耦设计

### 标准 OpenAI 接口 (`arena/env.py`)
`LLMEnv` 是**唯一**的大模型访问入口，使用 `openai.OpenAI` 客户端：
- `env.generate(prompt, ...)` —— 黑盒文本生成（攻击改写、无水印基线、判官打分均走这里）。
- `env.get_transformers_config()` —— 当配置了本地模型时，为 MarkLLM 防御提供 logits 级访问。

防御 `generate(env, request)` 与攻击 `attack(env, task)` 都通过同一个 `env` 访问大模型，满足“大模型访问采用标准的 openai 接口”。

### 攻防解耦 (`arena/registry.py`)
- 攻击与防御各自注册到独立注册表（`register_attack` / `register_defense`）。
- 二者**互不引用**：新增攻击无需改动任何防御代码，反之亦然。
- `ConfrontationRunner.run_matrix()` 从两个注册表查询并构造完整的 Attack × Defense 矩阵。
- 防御始终是 `(generator, detector)` 绑定的 `DefenseBundle`，检测器密钥与生成器绑定，保证配对诚实。

### 复用 MarkLLM
- **防御**：`arena/defenses/markllm_defense.py` 通过 `AutoWatermark.load(...)` 封装 MarkLLM 全部 logits 水印算法（KGW、Unigram、SWEET、EXP、EXPEdit、ITSEdit、EXPGumbel、EWD、Unbiased、SynthID、PF、MorphMark）。私有 `secret_key` 注入为 MarkLLM 的 `hash_key`。
- **攻击**：`arena/attacks/markllm_attack.py` 封装 MarkLLM 的 `GPTParaphraser`、`SynonymSubstitution`、`WordDeletion`、`BackTranslationTextEditor` 等。

### 纯 API 实现（无需本地模型）
- **防御**：`BlackBoxGreenList` —— Best-of-N 绿名单水印，仅用 `env.generate()`，黑盒可嵌入并可检测。
- **攻击**：`LLMRewrite`、`LLMRoundTrip` —— 仅用 OpenAI 接口的改写/回译，破坏 token 级统计特征。

---

## 评测指标（对齐赛题）

| 指标 | 说明 | 赛题章节 |
|------|------|----------|
| ROC-AUC | 检测器对水印/非水印文本的区分能力 | 8.1 / 14.1 |
| TPR@FPR | 固定误报率下的检测率 | 8 |
| ASR | 攻击成功率（攻击后检测器失效比例） | 13.2 |
| 语义相似度 | 攻击前后文本语义保持（资格线 ≥0.85） | 9.2 / 13.1 |
| 文本质量保持率 | 水印文本与无水印文本相似度（资格线 ≥0.9） | 14.2 |
| 防御资格线 | AUC>0.8 且 质量保持率≥0.9 | 15 |
| 攻击资格线 | 语义相似度≥0.85 且 有效文本比例≥0.95 | 13.1 |
| 攻击最终得分 | 50%·ASR + 50%·文本保持 | 13.4 |

语义相似度后端可插拔：`sentence-transformers` → OpenAI embeddings → 词级 Jaccard 兜底（离线也可出分）。

---

## 已注册方法一览

**防御**（generator+detector）
| 名称 | 来源 | 需本地模型 |
|------|------|-----------|
| BlackBoxGreenList | API-native | 否 |
| KGW / Unigram / SWEET / EXP / EXPGumbel / EXPEdit / ITSEdit / EWD / Unbiased / SynthID / PF / MorphMark | MarkLLM | 是 |

**攻击**
| 名称 | 来源 | 用 LLM |
|------|------|--------|
| LLMRewrite | API-native | 是 |
| LLMRoundTrip | API-native | 是 |
| GPTParaphrase | MarkLLM | 是 |
| SynonymSubstitution / WordDeletion / BackTranslation | MarkLLM | 否 |
| Identity | API-native | 否 |

---

## 提交模板（赛题第二十一章）
- `submission/defense_template/`：`watermark.py` + `detector.py` + `config.yaml`，对齐赛题接口；`algorithm` 字段选择内置算法或自定义实现。
- `submission/attack_template/`：`attack.py` + `config.yaml`，`attack` 字段选择内置攻击或自定义实现。

---

## 备注
- logits 水印算法依赖本地 HuggingFace 模型；纯 OpenAI 端点无法修改 logits，此时请用 `BlackBoxGreenList` 或部署支持 logits 的本地服务（vLLM 等，其同样提供 OpenAI 兼容接口）。
- `download_datasets.py` 对每个在线源均做 best-effort，网络失败会跳过并继续，始终能产出可用数据集。
