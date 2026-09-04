<!-- markdown-translator:17fede1ac4dc205ebb445ab332288d700032b35d6456abb5ad37fdbdbb3364ff -->
<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Xiaomi-TabLDM: A Tabular Large Data Foundation Model<br/>For Classification and Regression via In-Context Learning
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/occams/Xiaomi-TabLDM" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://arxiv.org/abs" target="_blank">📔 Technical Report</a>
  &nbsp;|
  <a href="README_CN.md" target="_blank">中文</a>
  &nbsp;|
  English
  &nbsp;
</div>

<br/>

***

这个存储库是官方的实现**小米-TabLDM**.

表格基础模型建立了基于上下文学习的通用预测范式。将来自下游数据集的标记样本作为上下文，单个预训练模型可以直接进行预测，而无需特定于任务的训练。在此范例的基础上，我们引入了Xiaomi-TabLDM，这是一种通过上下文学习进行分类和回归的表格大数据基础模型，它无需针对特定任务进行微调即可提供卓越的预测精度。我们的模型专门针对结构因果模型 (SCM) 生成的合成数据进行预训练，可实现更灵活的上下文利用和更高效的容量扩展。

**新的绩效标准。** *跨基准的强大回归性能*：Xiaomi-TabLDM 在 OpenML-CTR23 上排名第一，在 TALENT、TabArena 和 BCCO 的回归上排名第二，在四个互补的基准套件中表现出持续强大的回归性能。*有利的性能-效率权衡*：Xiaomi-TabLDM 结合了强大的预测性能和大幅降低的计算成本。例如，在 TabArena 回归中，它实现了第二高的 Elo，同时比排名第一的 TabFM 减少了 82% 的训练时间和 68% 的预测时间。

**大规模综合预训练。**&#x5C0F;米-TabLDM扩展了用于预训练的合成表格数据的覆盖范围和多样性。我们还采用三阶段训练策略，结合双流特征分组、轻量级注意力残差和稀疏专家混合，使小米-TabLDM能够在不同的表格任务中学习更丰富的特征交互和专家专业化。

**测试时间缩放。**&#x5C0F;米-TabLDM通过测试时计算扩展进一步扩展了表格预测：在推理时分配额外的计算持续提高了基础模型的预测性能。

**便于使用：**&#x5C0F;米-TabLDM可以安装`pip`并提供了 scikit-learn 兼容的接口。`fit`不更新模型权重；它仅预处理上下文并加载预训练模型。预测是通过单次前向传播中的上下文学习产生的。

**快速地：**&#x901A;过KV缓存，重复调用`predict`在相同的训练数据上可以重用缓存的上下文投影，从而显着加速重复推理。对于较大的数据集，建议使用 GPU，并且可以使用 CPU/磁盘卸载来扩展到更大的数据大小。

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM.png"
    alt="Xiaomi-TabLDM"
    width="800"
  >
</div>

## 表现

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TALENT_Fig1.png"
    alt="Regression average-rank performance on TALENT (lower is better)"
    width="800"
  >
  <br>
  <em>Figure 1. Regression average-rank performance on TALENT (lower is better)</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TabArena_Fig1.png"
    alt=" Regression Elo performance on TabArena"
    width="800"
  >
  <br>
  <em>Figure 2. Regression Elo performance on TabArena (higher is better).</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_BCCO_Fig2.png"
    alt="Performance on BCCO"
    width="800"
  >
  <br>
  <em>Figure 3. Average-rank comparison on BCCO. Circles denote the average ranks on BCCO-CLS and BCCO-REG, while diamonds denote the overall average rank across the two settings. Models are ordered
by the combined average rank; lower is better. </em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM__OpenML-CTR23_Fig8.png"
    alt="Performance on OpenML-CTR23"
    width="800"
  >
  <br>
  <em>Figure 4. Average-rank comparison on OpenML-CTR23 over 33 regression datasets (lower is better).</em>
</div>

## 安装

```bash
cd xiaomi-tabldm
pip install .
```

根据需要安装可选的依赖项：

```bash
pip install .[numba]   # Optional JIT acceleration for the quantile distribution layer
pip install .[test]    # Test dependencies
```

安装 PyTorch`pip`在 Intel Mac 上可能会失败。如果是这样，请先安装 PyTorch：

```bash
conda install pytorch -c pytorch
```

### 依赖关系

`torch>=2.2`,`scikit-learn>=1.3.0`,`numpy`,`scipy`,`einops>=0.7`,`psutil`,`tqdm>=4.64.0`， 和`huggingface-hub`.`numba`是可选的。

## 基本用法

### 分类

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(model_path="checkpoints/clf_default.ckpt")
clf.fit(X_train, y_train)          # In-context learning: no weight updates
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

> `fit` **不训练模型**。它只预处理标记的上下文
> （`X_train`,`y_train`）并加载预训练的权重。执行预测
> 完全通过情境学习。首次使用时，会下载检查点
> 自动从拥抱面部中心。指定`model_path`使用本地文件
> 用于离线推理。

### KV缓存

打电话时`predict`多次使用相同的训练数据，例如在
评估，启用KV缓存可以避免重复计算上下文。缓存
建于`fit`并在后续过程中重复使用`predict`来电。请注意，这
需要额外的 GPU/CPU 内存，因此请根据您的使用案例选择设置：

> 超过10类的分类任务不支持KV缓存。
> 保持`kv_cache=False`（默认）这些数据集；否则`fit`提出一个
> 错误。

```python
clf = TabLDMClassifier(
    kv_cache=True, model_path="checkpoints/clf_default.ckpt"
)
clf.fit(X_train, y_train)          # Build the cache once
clf.predict(X_test_batch_1)        # Reuse the cached context
clf.predict(X_test_batch_2)
```

### 保存/加载

```python
clf.save(
    "classifier.pkl",
    save_model_weights=False,  # If False, reload weights from the checkpoint
    save_training_data=True,   # If True, include training data; False improves privacy
    save_kv_cache=True,        # Save the KV cache when available
)

from tabldm import TabLDMClassifier
clf = TabLDMClassifier.load("classifier.pkl")
```

什么时候`save_model_weights=False`（默认），保存的文件较小，但
重量必须重新加载`model_path`或加载估算器时的集线器。

## 高级配置

小米-TabLDM提供了一组用于定制推理行为的参数。以下
示例显示所有可用的分类器参数及其默认值：

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(
    n_estimators=8,               # Ensemble members; more is more accurate but slower
    norm_methods=None,            # Normalization methods to try
    feat_shuffle_method="latin",  # Feature permutation strategy
    class_shuffle_method="shift", # Class permutation strategy
    outlier_threshold=4.0,        # Z-score threshold for outlier detection/clipping
    softmax_temperature=0.9,      # Temperature controlling prediction confidence
    average_logits=True,          # Average logits (True) or probabilities (False)
    support_many_classes=True,    # Automatically handle more than 10 classes
    batch_size=8,                 # Ensemble members processed together; lower saves memory
    kv_cache=False,               # Cache training-data KV projections for repeated inference
    model_path=None,              # Checkpoint path; None downloads from Hugging Face
    allow_auto_download=True,     # Download automatically when not found locally
    checkpoint_version="checkpoints/clf_default.ckpt",  # Pretrained checkpoint version
    device=None,                  # Inference device; None selects CUDA or CPU automatically
    use_amp="auto",               # Automatic mixed precision for faster inference
    use_fa3="auto",               # Flash Attention 3 on Hopper GPUs such as H100
    offload_mode="auto",          # Decide automatically when to use CPU/disk offloading
    disk_offload_dir=None,        # Directory for disk offloading
    random_state=42,              # Random seed for reproducibility
    n_jobs=None,                  # Number of PyTorch threads for CPU inference
    verbose=False,                # Print detailed inference information
    inference_config=None,        # Fine-grained inference control for advanced users
)
```

`TabLDMRegressor`除了特定于分类的参数外，接受相同的参数
参数`class_shuffle_method`,`softmax_temperature`,`average_logits`， 和`support_many_classes`.

## 加载检查点

检查点按以下顺序解决：

1. **`model_path`**— 如果它指向现有文件，则直接使用该文件。
2. 如果`model_path`已设置但文件不存在并且`allow_auto_download=True`,
   名为的检查点`checkpoint_version`被下载到`model_path`.
3. 如果`model_path`是`None`，从 Hugging Face Hub 检索检查点
   缓存使用`checkpoint_version`作为钥匙。

这`checkpoint_version`value 是 Hugging Face 存储库中的文件名，而不是
本地文件系统路径。第一次查找使用本地 Hugging Face 缓存
（通常`~/.cache/huggingface/hub`）；如果文件没有被缓存并且`allow_auto_download=True`，它会自动下载。

对于完全离线推理，点`model_path`到本地文件：

```python
clf = TabLDMClassifier(
    model_path="checkpoints/clf_default.ckpt", allow_auto_download=False
)
```

## 可用型号

|型号|分类器|回归器 |
| -------------- | -------------------- | ------------------- |
|**小米-TabLDM**|[`XiaomiTabLDMClassifier`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/clf_default.ckpt)|[`XiaomiTabLDMRegressor`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/reg_default.ckpt)|

### 例子

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

默认情况下，测试会在以下位置查找检查点`../checkpoints`。覆盖
这个位置与`TABLDM_CKPT_DIR`环境变量。如果没有检查点
发现，测试会自动跳过。

## 执照

该项目是在[阿帕奇许可证 2.0](LICENSE).

版权所有(C)2026 小米公司

## FAQ

**什么是小米-TabLDM？**&#x5C0F;米-TabLDM是一个类似于TabPFN和TabICL的表格基础模型。它学习新数据
通过预训练 Transformer 的单次前向传递中的上下文学习：`y_pred = model(X_train, y_train, X_test)`（由内部调用`predict()`）。
它的学习能力来自于大规模合成数据的预训练。

**小米-TabLDM 有多快？**&#x5BF9;于具有 $n$ 训练行和 $m$ 列的数据集，运行时复杂度为
$O(n^2 + nm^2)$。 KV缓存加速对相同训练数据的重复推理，
而 CPU/磁盘卸载可以在不运行的情况下处理更大的数据集
内存不足。

**什么数据集大小合适？**&#x9884;训练数据涵盖数百到数万个训练样本
数据集范围从几个到一百多个特征列。该模型可以
推断到更大的范围，尽管随着数据超出范围，准确性可能会下降
训练分布。具体推荐范围将在经验后添加
评价。

## 预处理

### 内置预处理

为了`X`，Xiaomi-TabLDM 接受 pandas DataFrame 或 NumPy 数组并执行
以下操作：

* 检测和序数编码分类列，包括字符串、对象、类别、
  和布尔列。在 NumPy 数组中，所有列共享相同的数据类型，并且
  整数列被视为数字。
* 为分类特征中的缺失值创建单独的类别。
* 均值插补缺失数值，编码为 NaN。
* 检测并剪裁异常值。
* 缩放和标准化特征。
* 排列特征以增加整体多样性。

## 封装布局

```
tabldm/
├── __init__.py          # Public API: estimators + InferenceConfig
├── __about__.py         # Version number
├── _model/              # PyTorch model + inference engine
│   ├── tabldm.py                 # Base TabLDM module
│   ├── attnres_light_rmsnorm.py # AttnRes/RMSNorm architecture
│   ├── attnres_light_rmsnorm_moe.py # MoE architecture
│   ├── embedding*.py, interaction.py, learning.py, encoders.py, layers.py
│   ├── attention.py, rope.py, ssmax.py, moe.py, quantile_dist.py
│   ├── kv_cache.py, kv_cache_attnres.py
│   ├── inference.py, inference_config.py
└── _sklearn/            # scikit-learn interface
    ├── base.py, classifier.py, regressor.py
    ├── preprocessing.py, sklearn_utils.py
    └── *_dualstream_moe.py   # MoE estimators
```

## 引文

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
