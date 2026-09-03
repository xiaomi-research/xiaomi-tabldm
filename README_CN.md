<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Xiaomi-TabLDM：一种基于上下文学习的表格大型数据基础模型，用于分类与回归任务
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<br/>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/occams/Xiaomi-TabLDM" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://arxiv.org/abs" target="_blank">📔 Technical Report</a>
  &nbsp;|
  中文
  &nbsp;|
  <a href="README.md" target="_blank">English</a>
  &nbsp;
</div>

<br/>

---

本仓库是 **Xiaomi-TabLDM** 的官方实现，包含表格基础模型 **TabLDM**。

表格基础模型建立了基于上下文学习（in-context learning）的通用预测范式。给定下游数据集中的带标签样本作为上下文，单个预训练模型即可直接进行预测，无需针对特定任务进行训练。在此基础上，我们提出了 TabLDM，一种新的表格基础模型，能够实现更灵活的上下文利用和更高效的模型容量扩展。

**新的性能标准。** 在具有挑战性的 TALENT 基准上，TabLDM 在二分类任务中优于所有基线，整体排名第二。在 TabArena 上，TabLDM 超越传统机器学习基线，整体排名第三，并在回归任务中排名第二。值得注意的是，在相近的模型规模下，TabLDM 优于 TabPFN-3；同时仅使用显著更少的总参数量和激活参数量，性能仍接近 TabFM。

**大规模合成预训练。** TabLDM 扩展了预训练所使用的合成表格数据的覆盖范围和多样性。我们还改进了三阶段训练策略，逐步引入双流特征组（dual-stream feature groups）、轻量级注意力残差（lightweight Attention Residual）和稀疏混合专家（sparse Mixture-of-Experts）。这些组件使 TabLDM 能够学习更丰富的特征交互，并在多样化的表格任务中实现专家专业化。

**测试时扩展。** TabLDM 进一步通过测试时计算扩展提升表格预测性能：在推理阶段分配额外计算资源，可以相较于基础模型持续提升预测表现。

**易于使用：** TabLDM 可通过 `pip` 安装，并提供与 scikit-learn 兼容的接口。
`fit` 不会更新模型权重，仅对上下文进行预处理并加载预训练模型。
预测通过上下文学习在单次前向传播中完成。

**速度快：** 通过 KV 缓存，在同一份训练数据上重复调用 `predict` 时可以复用缓存的上下文投影，从而显著加速重复推理。对于较大的数据集，建议使用 GPU；同时可以通过 CPU / 磁盘卸载扩展到更大的数据规模。

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM.png"
    alt="Xiaomi-TabLDM"
    width="800"
  >
</div>

## 性能

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TALENT_Fig1.png"
    alt="TALENT 上的整体排名"
    width="800"
  >
  <br>
  <em>图 1. TALENT 上的整体排名。</em>
</div>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TabArena_Fig1.png"
    alt="TabArena 上的回归 Elo 性能"
    width="800"
  >
  <br>
  <em>图 2. TabArena 上的回归 Elo 性能。</em>
</div>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_BCCO_Fig2.png"
    alt="BCCO 上的性能"
    width="800"
  >
  <br>
  <em>图 3. BCCO 上的平均排名对比。圆形表示 BCCO-CLS 和 BCCO-REG 上的平均排名，菱形表示两个设置下的整体平均排名。</em>
</div>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM__OpenML-CTR23_Fig8.png"
    alt="OpenML-CTR23 上的性能"
    width="800"
  >
  <br>
  <em>图 4. OpenML-CTR23 上 33 个回归数据集的平均排名对比。</em>
</div>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TrainingEfficiency_Fig10.png"
    alt="训练效率"
    width="800"
  >
  <br>
  <em>图 5. 不同任务类型下训练效率与可改进性之间的权衡：（左）分类性能，（右）整体任务。</em>
</div>

## 安装

```bash
cd xiaomi-tabldm
pip install .
```

可按需安装可选依赖：

```bash
pip install .[numba]   # 分位数分布层的可选 JIT 加速
pip install .[test]    # 测试依赖
```

在 Intel Mac 上通过 `pip` 安装 PyTorch 可能失败。如果遇到此问题，请先安装 PyTorch：

```bash
conda install pytorch -c pytorch
```

然后按照上述方式安装 `tabldm`。

### 依赖

`torch>=2.2`、`scikit-learn>=1.3.0`、`numpy`、`scipy`、`einops>=0.7`、
`psutil`、`tqdm>=4.64.0` 和 `huggingface-hub`。`numba` 为可选依赖。

## 基本用法

### 分类

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(model_path="checkpoints/clf_default.ckpt")
clf.fit(X_train, y_train)          # 上下文学习：不更新权重
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)  # (n_test, n_classes)
```

### 回归

```python
from tabldm import TabLDMRegressor

reg = TabLDMRegressor(model_path="checkpoints/reg_default.ckpt")
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

> `fit` **不会训练模型**。它仅对带标签的上下文（`X_train`、`y_train`）进行预处理并加载预训练权重，预测完全通过上下文学习完成。首次使用时，checkpoint 会自动从 Hugging Face Hub 下载。若需离线推理，可通过 `model_path` 指定本地文件。

### KV 缓存

当需要在同一份训练数据上多次调用 `predict`（例如进行评测）时，启用 KV 缓存可以避免重复计算上下文。缓存会在 `fit` 时构建，并在后续的 `predict` 调用中复用。注意，KV 缓存需要额外的 GPU / CPU 内存，请根据实际使用场景选择配置：

> 类别数超过 10 的分类任务不支持 KV 缓存。对于此类数据集，请保持 `kv_cache=False`（默认值），否则 `fit` 会报错。

```python
clf = TabLDMClassifier(
    kv_cache=True, model_path="checkpoints/clf_default.ckpt"
)
clf.fit(X_train, y_train)          # 一次性构建缓存
clf.predict(X_test_batch_1)        # 复用缓存的上下文
clf.predict(X_test_batch_2)
```

### 保存 / 加载

```python
clf.save(
    "classifier.pkl",
    save_model_weights=False,  # False：加载时从 checkpoint 重新读取权重
    save_training_data=True,   # True：包含训练数据；False：有助于保护数据隐私
    save_kv_cache=True,        # 如果存在 KV 缓存，则一并保存
)

from tabldm import TabLDMClassifier
clf = TabLDMClassifier.load("classifier.pkl")
```

当 `save_model_weights=False`（默认）时，保存的文件更小，但加载估算器时需要从 `model_path` 或 Hugging Face Hub 重新加载权重。

## 高级配置

TabLDM 提供了一组用于自定义推理行为的参数。以下示例展示分类器的全部可用参数及其默认值：

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(
    n_estimators=8,               # 集成成员数，更多通常更准确但速度更慢
    norm_methods=None,            # 尝试的归一化方法
    feat_shuffle_method="latin",  # 特征排列策略
    class_shuffle_method="shift", # 类别排列策略
    outlier_threshold=4.0,        # 离群点检测 / 截断的 z-score 阈值
    softmax_temperature=0.9,      # 控制预测置信度的温度
    average_logits=True,          # 平均 logits（True）或概率（False）
    support_many_classes=True,    # 自动处理超过 10 类的任务
    batch_size=8,                 # 同时处理的集成成员数，降低可节省内存
    kv_cache=False,               # 缓存训练数据 KV 投影，加速重复推理
    model_path=None,              # checkpoint 路径；None 则从 Hugging Face 下载
    allow_auto_download=True,     # 本地不存在时自动下载
    checkpoint_version="checkpoints/clf_default.ckpt",  # 预训练 checkpoint 版本
    device=None,                  # 推理设备；None 则自动选择 CUDA 或 CPU
    use_amp="auto",               # 自动混合精度，加速推理
    use_fa3="auto",               # Hopper GPU（如 H100）的 Flash Attention 3
    offload_mode="auto",          # 自动决定何时使用 CPU / 磁盘卸载
    disk_offload_dir=None,        # 磁盘卸载目录
    random_state=42,              # 保证可复现的随机种子
    n_jobs=None,                  # CPU 推理使用的 PyTorch 线程数
    verbose=False,                # 打印详细的推理信息
    inference_config=None,        # 面向高级用户的细粒度推理控制
)
```

`TabLDMRegressor` 接受相同的参数，但不包含分类专属参数：
`class_shuffle_method`、`softmax_temperature`、`average_logits` 和
`support_many_classes`。

## 加载 checkpoint

checkpoint 按以下顺序解析：

1. **`model_path`** —— 如果它指向一个已存在的文件，则直接使用该文件。
2. 如果设置了 `model_path`，但文件不存在，并且 `allow_auto_download=True`，则下载 `checkpoint_version` 指定的 checkpoint 到 `model_path`。
3. 如果 `model_path` 为 `None`，则使用 `checkpoint_version` 作为键，从 Hugging Face Hub 缓存中获取 checkpoint。



`checkpoint_version` 表示 Hugging Face 仓库中的文件名，而不是本地文件系统路径。程序会首先从本地 Hugging Face 缓存中查找（通常位于 `~/.cache/huggingface/hub`）；如果缓存中不存在该文件，并且 `allow_auto_download=True`，则会自动下载。

若需完全离线推理，将 `model_path` 指向本地文件：

```python
clf = TabLDMClassifier(
    model_path="checkpoints/clf_default.ckpt", allow_auto_download=False
)
```

## 可用模型

| 模型 | 分类估算器 | 回归估算器 |
| --- | --- | --- |
| **TabLDM** | [`TabLDMClassifier`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/clf_default.ckpt) | [`TabLDMRegressor`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/reg_default.ckpt) |

### 示例

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(
    model_path="checkpoints/clf_default.ckpt",
    device="cuda",
)
clf.fit(X_train, y_train)
clf.predict(X_test)
```

对应的回归估算器为 `TabLDMRegressor`。

## 测试

```bash
cd xiaomi-tabldm
pytest tests/test_infer_package.py -v
```

测试默认在 `../checkpoints` 中查找 checkpoint。可以通过环境变量
`TABLDM_CKPT_DIR` 覆盖该位置。如果未找到 checkpoint，测试会自动跳过。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 发布。

Copyright (C) 2026 Xiaomi Corporation

## 常见问题

**什么是 TabLDM？**

TabLDM 是一个类似于 TabPFN 和 TabICL 的表格基础模型。它通过上下文学习，在预训练 Transformer 的单次前向传播中学习新数据：
`y_pred = model(X_train, y_train, X_test)`（在 `predict()` 内部调用）。
它的学习能力来自在大规模合成数据上的预训练。

**TabLDM 有多快？**

对于包含 $n$ 条训练数据和 $m$ 列的数据集，运行时复杂度为 $O(n^2 + nm^2)$。KV 缓存可以加速同一训练数据上的重复推理；CPU / 磁盘卸载则支持在不发生内存溢出的情况下处理更大的数据集。

**适合使用什么规模的数据集？**

预训练数据覆盖数百至数万条训练样本，以及从少量到一百多个特征列的数据集。模型可以外推到更大的规模，但当数据超出训练分布时，准确率可能下降。具体推荐范围将在完成实证评估后补充。

## 预处理

### 内置预处理

对于 `X`，TabLDM 接受 pandas DataFrame 或 NumPy 数组，并执行以下操作：

- 检测并对类别列进行 ordinal 编码，包括 string、object、category 和 boolean 类型。对于 NumPy 数组，所有列共享相同的数据类型，整型列会被视为数值列。
- 为类别特征中的缺失值创建单独的类别。
- 对编码为 NaN 的缺失数值执行均值插补。
- 检测并截断离群点。
- 对特征进行缩放和归一化。
- 对特征进行排列，以增加集成多样性。

## 包结构

```
tabldm/
├── __init__.py          # 公共 API：估算器 + InferenceConfig
├── __about__.py         # 版本号
├── _model/              # PyTorch 模型 + 推理引擎
│   ├── tabldm.py                 # 基础 TabLDM 模块
│   ├── attnres_light_rmsnorm.py # AttnRes / RMSNorm 架构
│   ├── attnres_light_rmsnorm_moe.py # MoE 架构
│   ├── embedding*.py, interaction.py, learning.py, encoders.py, layers.py
│   ├── attention.py, rope.py, ssmax.py, moe.py, quantile_dist.py
│   ├── kv_cache.py, kv_cache_attnres.py
│   ├── inference.py, inference_config.py
└── _sklearn/            # scikit-learn 接口
    ├── base.py, classifier.py, regressor.py
    ├── preprocessing.py, sklearn_utils.py
    └── *_dualstream_moe.py   # MoE 估算器
```

## Citation

```bash
@article{wang2026xiaomitabldm,
  title         = {{Xiaomi-TabLDM}: Technical Report},
  author        = {Penghui Wang and Wei Liu and Hong Wang and Chengyue Huang and Yuxi Sun and Zirui Wang and Hongming Huang and Zhenwei Xin and Chunxiao Liu and Erli Meng and Bin Wang},
  year          = {2026},
  eprint        = {2609.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

```
