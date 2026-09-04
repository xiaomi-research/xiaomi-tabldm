<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Xiaomi-TabLDM：一种基于上下文学习、用于分类与回归的表格大数据基础模型
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/occams/Xiaomi-TabLDM" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://arxiv.org/abs/2609.03880" target="_blank">📔 技术报告</a>
  &nbsp;|
  中文
  &nbsp;|
  <a href="README.md" target="_blank">English</a>
  &nbsp;
</div>

<br/>

---

本仓库是 **Xiaomi-TabLDM** 的官方实现。

表格基础模型通过上下文学习建立了一种通用的预测范式：将下游数据集中的带标签样本作为上下文输入后，单个预训练模型无需针对具体任务进行训练即可直接完成预测。在此范式的基础上，我们提出了 Xiaomi-TabLDM——一种通过上下文学习进行分类和回归的表格大数据基础模型。该模型无需针对具体任务进行微调，即可实现出色的预测精度。Xiaomi-TabLDM 仅使用结构因果模型（Structural Causal Models，SCMs）生成的合成数据进行预训练，从而能够更灵活地利用上下文，并更高效地扩展模型容量。

**树立新的性能标杆。** _跨基准测试展现出强劲的回归性能_：Xiaomi-TabLDM 在 OpenML-CTR23 上排名第一，并在 TALENT、TabArena 和 BCCO 的回归任务中排名第二，在四个互补的基准测试套件中均展现出稳定而强劲的回归性能。_实现良好的性能—效率权衡_：Xiaomi-TabLDM 在保持强大预测性能的同时，显著降低了计算成本。例如，在 TabArena 回归任务中，该模型取得了第二高的 Elo 分数，但相比排名第一的 TabFM，训练时间减少了 82%，预测时间减少了 68%。

**大规模合成数据预训练。** Xiaomi-TabLDM 扩展了预训练所使用的合成表格数据的覆盖范围和多样性。我们还采用三阶段训练策略，并结合双流特征分组、轻量级 Attention Residual 和稀疏混合专家（Mixture-of-Experts，MoE），使 Xiaomi-TabLDM 能够在多样化的表格任务中学习更丰富的特征交互关系，并实现专家分工。

**测试时扩展。** Xiaomi-TabLDM 通过测试时计算扩展进一步增强表格预测能力：在推理阶段分配更多计算资源，可以持续提升模型相对于基础版本的预测性能。

**易于使用：** Xiaomi-TabLDM 可通过 `pip` 安装，并提供兼容 scikit-learn 的接口。`fit` 不会更新模型权重，而只负责预处理上下文并加载预训练模型。预测完全通过一次前向传播中的上下文学习完成。

**速度快：** 启用 KV 缓存后，在相同训练数据上重复调用 `predict` 时可以复用已缓存的上下文投影，从而显著加快重复推理。对于较大的数据集，建议使用 GPU；同时也可以通过 CPU/磁盘卸载支持更大规模的数据。

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
    alt="TALENT 上的回归平均排名性能（越低越好）"
    width="800"
  >
  <br>
  <em>图 1：TALENT 上的回归平均排名性能（越低越好）</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TabArena_Fig1.png"
    alt="TabArena 上的回归 Elo 性能"
    width="800"
  >
  <br>
  <em>图 2：TabArena 上的回归 Elo 性能（越高越好）</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_BCCO_Fig2.png"
    alt="BCCO 上的性能"
    width="800"
  >
  <br>
  <em>图 3：BCCO 上的平均排名对比。圆点表示 BCCO-CLS 和 BCCO-REG 两种设置下的平均排名，菱形表示两种设置的总体平均排名。模型按照合并后的平均排名排序；排名越低越好。</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM__OpenML-CTR23_Fig8.png"
    alt="OpenML-CTR23 上的性能"
    width="800"
  >
  <br>
  <em>图 4：OpenML-CTR23 上 33 个回归数据集的平均排名对比（越低越好）</em>
</div>

## 安装

```bash
cd xiaomi-tabldm
pip install .
```

根据需要安装可选依赖：

```bash
pip install .[numba]   # 为分位数分布层提供可选的 JIT 加速
pip install .[test]    # 测试依赖
```

在 Intel Mac 上使用 `pip` 安装 PyTorch 可能会失败。如果遇到此问题，请先安装 PyTorch：

```bash
conda install pytorch -c pytorch
```

### 依赖

`torch>=2.2`、`scikit-learn>=1.3.0`、`numpy`、`scipy`、`einops>=0.7`、`psutil`、`tqdm>=4.64.0` 和 `huggingface-hub`。`numba` 为可选依赖。

## 基本用法

### 分类

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(model_path="checkpoints/clf_default.ckpt")
clf.fit(X_train, y_train)          # 上下文学习：不会更新模型权重
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

> `fit` **不会训练模型**，只会预处理带标签的上下文（`X_train`、`y_train`）并加载预训练权重。预测完全通过上下文学习完成。首次使用时，检查点会自动从 Hugging Face Hub 下载。若要进行离线推理，请指定本地文件作为 `model_path`。

### KV 缓存

在评估等场景中，如果需要使用相同的训练数据多次调用 `predict`，启用 KV 缓存可以避免重复计算上下文。缓存会在 `fit` 期间构建，并在后续的 `predict` 调用中复用。请注意，KV 缓存会额外占用 GPU/CPU 内存，请根据实际使用场景进行选择：

> 对于类别数超过 10 的分类任务，不支持 KV 缓存。对于此类数据集，请保持 `kv_cache=False`（默认值），否则 `fit` 会抛出错误。

```python
clf = TabLDMClassifier(
    kv_cache=True, model_path="checkpoints/clf_default.ckpt"
)
clf.fit(X_train, y_train)          # 仅需构建一次缓存
clf.predict(X_test_batch_1)        # 复用已缓存的上下文
clf.predict(X_test_batch_2)
```

### 保存与加载

```python
clf.save(
    "classifier.pkl",
    save_model_weights=False,  # 为 False 时，从检查点重新加载权重
    save_training_data=True,   # 为 True 时包含训练数据；设为 False 可提高隐私性
    save_kv_cache=True,        # 在可用时保存 KV 缓存
)

from tabldm import TabLDMClassifier
clf = TabLDMClassifier.load("classifier.pkl")
```

当 `save_model_weights=False`（默认值）时，保存的文件更小，但加载估计器时必须从 `model_path` 或 Hugging Face Hub 重新加载权重。

## 高级配置

Xiaomi-TabLDM 提供了一组用于自定义推理行为的参数。以下示例展示了分类器支持的全部参数及其默认值：

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(
    n_estimators=8,               # 集成成员数量；数量越多通常越准确，但速度越慢
    norm_methods=None,            # 尝试使用的归一化方法
    feat_shuffle_method="latin",  # 特征置换策略
    class_shuffle_method="shift", # 类别置换策略
    outlier_threshold=4.0,        # 异常值检测/裁剪的 Z 分数阈值
    softmax_temperature=0.9,      # 控制预测置信度的温度参数
    average_logits=True,          # 对 logits（True）或概率（False）取平均
    support_many_classes=True,    # 自动处理类别数超过 10 的任务
    batch_size=8,                 # 一次处理的集成成员数；调低可节省内存
    kv_cache=False,               # 缓存训练数据的 KV 投影，以加速重复推理
    model_path=None,              # 检查点路径；为 None 时从 Hugging Face 下载
    allow_auto_download=True,     # 本地未找到时自动下载
    checkpoint_version="checkpoints/clf_default.ckpt",  # 预训练检查点版本
    device=None,                  # 推理设备；为 None 时自动选择 CUDA 或 CPU
    use_amp="auto",               # 自动混合精度，可加快推理
    use_fa3="auto",               # 在 H100 等 Hopper GPU 上使用 Flash Attention 3
    offload_mode="auto",          # 自动决定是否使用 CPU/磁盘卸载
    disk_offload_dir=None,        # 磁盘卸载目录
    random_state=42,              # 用于保证可复现性的随机种子
    n_jobs=None,                  # CPU 推理使用的 PyTorch 线程数
    verbose=False,                # 输出详细的推理信息
    inference_config=None,        # 面向高级用户的细粒度推理控制
)
```

除分类专用参数 `class_shuffle_method`、`softmax_temperature`、`average_logits` 和 `support_many_classes` 外，`TabLDMRegressor` 接受相同的参数。

## 加载检查点

检查点按以下顺序解析：

1. **`model_path`**：如果它指向一个已存在的文件，则直接使用该文件。
2. 如果设置了 `model_path`，但对应文件不存在，且 `allow_auto_download=True`，则将 `checkpoint_version` 指定的检查点下载到 `model_path`。
3. 如果 `model_path` 为 `None`，则使用 `checkpoint_version` 作为键，从 Hugging Face Hub 缓存中检索检查点。

`checkpoint_version` 的值是 Hugging Face 仓库中的文件名，而不是本地文件系统路径。程序首先会查找本地 Hugging Face 缓存（通常位于 `~/.cache/huggingface/hub`）；如果文件尚未缓存且 `allow_auto_download=True`，则会自动下载。

如需进行完全离线的推理，请将 `model_path` 指向本地文件：

```python
clf = TabLDMClassifier(
    model_path="checkpoints/clf_default.ckpt", allow_auto_download=False
)
```

## 可用模型

| 模型 | 分类器 | 回归器 |
| --- | --- | --- |
| **Xiaomi-TabLDM** | [`XiaomiTabLDMClassifier`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/clf_default.ckpt) | [`XiaomiTabLDMRegressor`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/reg_default.ckpt) |

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

## 测试

```bash
cd xiaomi-tabldm
pytest tests/test_infer_package.py -v
```

默认情况下，测试会在 `../checkpoints` 中查找检查点。可以通过环境变量 `TABLDM_CKPT_DIR` 覆盖该位置。如果未找到检查点，测试会自动跳过。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。

版权所有 (C) 2026 小米公司

## 常见问题

**什么是 Xiaomi-TabLDM？**
Xiaomi-TabLDM 是一种类似于 TabPFN 和 TabICL 的表格基础模型。它通过预训练 Transformer 的单次前向传播完成上下文学习，从而学习新的数据：`y_pred = model(X_train, y_train, X_test)`（该过程由 `predict()` 在内部调用）。模型的学习能力来自大规模合成数据预训练。

**Xiaomi-TabLDM 的速度有多快？**
对于包含 $n$ 行训练数据和 $m$ 列特征的数据集，运行时复杂度为 $O(n^2 + nm^2)$。KV 缓存可以加速相同训练数据上的重复推理，而 CPU/磁盘卸载则能够在避免内存溢出的情况下处理更大的数据集。

**适合使用多大规模的数据集？**
预训练数据覆盖数百至数万个训练样本，数据集的特征列数从几个到一百多个不等。模型可以外推到更大的规模，但当数据超出训练分布后，准确率可能会下降。具体的推荐范围将在完成实证评估后补充。

## 预处理

### 内置预处理

对于 `X`，Xiaomi-TabLDM 接受 pandas DataFrame 或 NumPy 数组，并执行以下操作：

- 检测并对分类列进行序数编码，包括字符串、对象、类别和布尔列。在 NumPy 数组中，所有列共享相同的数据类型，整数列会被视为数值列。
- 为分类特征中的缺失值创建单独的类别。
- 对编码为 NaN 的缺失数值进行均值填补。
- 检测并裁剪异常值。
- 对特征进行缩放和归一化。
- 对特征进行置换，以增加集成多样性。

## 包结构

```
tabldm/
├── __init__.py          # 公共 API：估计器与 InferenceConfig
├── __about__.py         # 版本号
├── _model/              # PyTorch 模型与推理引擎
│   ├── tabldm.py                 # TabLDM 基础模块
│   ├── attnres_light_rmsnorm.py # AttnRes/RMSNorm 架构
│   ├── attnres_light_rmsnorm_moe.py # MoE 架构
│   ├── embedding*.py, interaction.py, learning.py, encoders.py, layers.py
│   ├── attention.py, rope.py, ssmax.py, moe.py, quantile_dist.py
│   ├── kv_cache.py, kv_cache_attnres.py
│   ├── inference.py, inference_config.py
└── _sklearn/            # scikit-learn 接口
    ├── base.py, classifier.py, regressor.py
    ├── preprocessing.py, sklearn_utils.py
    └── *_dualstream_moe.py   # MoE 估计器
```

## 引用

```bibtex
@article{wang2026xiaomitabldm,
  title         = {{Xiaomi-TabLDM}: Technical Report},
  author        = {Penghui Wang and Wei Liu and Hong Wang and Chengyue Huang and Yuxi Sun and Zirui Wang and Hongming Huang and Zhenwei Xin and Chunxiao Liu and Erli Meng and Bin Wang},
  year          = {2026},
  eprint        = {2609.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  =  {cs.AI}
}
```
