# LimiX Pipeline + 单次验证 NNLS 改造方案

## 已确认的决策

- 完整替换 `enhance_candidates=True` 的当前增强实现；非增强路径保持不变。
- 默认成员直接采用 LimiX V2 原始默认 JSON：分类 32 条、回归 8 条。
- `fit()` 内部仅做一次固定 20% holdout：分类使用分层拆分，回归使用随机拆分；不保留 K-fold/OOF 路由。
- 使用 SciPy `nnls` 拟合非负权重，并归一化为和为 1 的推理权重。
- 回归按 LimiX 原始 `reg_default_noretrieval_v2.json`：前四条 `quantile_uniform_all_data + ordinal_strict_feature_shuffled + shuffle + SVD`，后四条 `power + onehot + shuffle`。

## 1. 新增声明式 pipeline 基础设施

在 `tabldm/_sklearn/preprocessing.py` 中实现/导出：

1. 不可变 `PipelineSpec` 描述单个成员：`name`、`numeric_transform`（映射 LimiX worker tag）、`discrete_flag`、`original_flag`、`categorical_encoding`、`feature_shuffle`、`fingerprint`、`max_interactions`、`svd_components`、`seed_offset`；回归额外支持 `target_transform`，但 LimiX 默认成员均为 `None`。
2. 两组常量工厂：`default_classifier_pipeline_specs()` 精确翻译 LimiX 分类 JSON 的 32 项，`default_regressor_pipeline_specs()` 精确翻译 LimiX 回归 JSON 的 8 项；不导入外部 LimiX 代码或依赖外部绝对路径。
3. `PipelineMember`：按固定顺序保存训练得到的变换状态，并提供 `fit(X, y)` / `transform(X)`：
   - 交互特征（可选） → `UniqueFeatureFilter` → 数值分布转换（含 `quantile_*`、`power`、`robust`、`kdi_uni`、无转换）与可选原始列拼接 → 类别编码（ordinal 家族、one-hot、numeric/none） → fingerprint（可选） → SVD 追加成分（可选） → 固定列置换。
   - 所有随机操作由 `random_state + seed_offset` 决定；训练后所有编码器、列掩码、SVD、哈希 salt、交互对和列置换均保存在成员上。
   - 保证输出为 TabLDM 现有前向函数可接收的稠密、有限 `float` 二维矩阵；one-hot 必须使用训练时学习到的类别和训练列宽度。
4. `PipelineEnsemble`：顺序持有成员，分别在完整训练集和 holdout 训练部分构造同一份 spec 列表；提供有序的成员名称、逐成员训练/测试矩阵和失败审计记录。成员编号永远来自 spec 顺序，不再依赖 group/奇偶编号。

## 2. 分类 estimator 的完整替换

在 `tabldm/_sklearn/classifier.py`：

1. 将增强相关构造参数收敛为 `pipeline_specs=None`、`validation_size=0.2`、`validation=True`；保留旧参数为兼容属性但在 `enhance_candidates=True` 时不再驱动 pipeline 选择。删除/停用 adaptive routing、quantile/SVD/adaptive-plus/Gaussian-rank group 的构建和 `n_estimators==32` 特判。
2. 在特征初始数值化、标签编码后，构建 `PipelineEnsemble(classification=True, specs=provided_or_default, ...)`。
3. 单次验证流程：
   - 分类用 `train_test_split(..., test_size=validation_size, stratify=y, random_state=random_state)`；若无法分层（最小类少于 2 或拆分失败），记录原因并让全部成功成员等权。
   - 只在 holdout-train 子集分别拟合 32 条成员，逐成员把 `(X_train_view, y_train_permuted, X_val_view)` 交给现有 TabLDM forward helper；每条成员若有类别标签置换，训练标签应用置换，预测概率立即逆映射到 canonical `classes_` 顺序。
   - 丢弃无法产生有限、非负且正确形状概率的 holdout 成员，并记录失败原因；对剩余成员的 `(E, n_val, n_classes)` 概率矩阵解 NNLS：`A = probs.reshape(E, -1).T`，`b = one_hot(y_val).ravel()`；归一化正权重。若无有效成员、求解异常或总权重为零，使用有效成员等权。
   - 将验证成员丢弃掩码映射到全训练 members：只有 holdout 成功的成员才能进入最终集成，以保证 NNLS 索引一一对应。
4. 在完整数据上重新拟合仅保留的成员；`predict_proba` 对每个成员生成 canonical 概率并按 `nnls_weights_` 加权，最后进行行归一化；`predict` 保持对 `classes_` 的 argmax 映射。
5. 默认关闭/移除现有基于 OOF 概率的 calibration 路径；避免将已被删除的 `_cal_P_` / `_cal_y_` 状态用于校准。`enable_calibration=True` 明确提示该模式在新 pipeline ensemble 中未启用，或直接把该参数默认改为 False（实施时选择不静默执行旧逻辑）。

## 3. 回归 estimator 的完整替换

在 `tabldm/_sklearn/regressor.py`：

1. 同样以 `pipeline_specs=None`、`validation_size=0.2` 和 `validation=True` 作为增强配置；删除/停用当前主/quantile/HK/交叉/SVD group、自适应 routing、K-fold 分支和 `foundation_rate` 组合逻辑在增强路径中的作用。
2. 保存完整训练目标 scaler；一次 holdout 用 `train_test_split(..., test_size=validation_size, shuffle=True, random_state=random_state)`。
3. 在 holdout-train 上分别拟合 8 条回归 pipeline：成员输入只使用其内部 `fit` 状态；模型 target 输入使用 estimator 的标准化目标或成员 target transform 后的目标。每条验证预测都在求 NNLS 前逆变换到原始 `y` 尺度。
4. 过滤非有限或形状错误成员，以 `(n_val, E)` 和原始尺度 `y_val` 调用 `scipy.optimize.nnls`；将正权重归一化为 `nnls_weights_`，求解失败时有效成员等权。将成功成员索引保存为 `nnls_valid_member_indices_`。
5. 在完整 X/y 上重新拟合成功成员，预测时每个成员的标量预测先逆目标变换、再逆 `y_scaler_` 回原始尺度，按已学习权重相加；不再调用旧 group-based `predict` 分支。

## 4. 公共行为与边界

- `enhance_candidates=False` 保持当前 `EnsembleGenerator` 路径、KV cache 行为和 public sklearn API 不变。
- `enhance_candidates=True` 继续禁止 KV cache。
- 特征输入仍通过 `TransformToNumerical`；它产生的 `categorical_indices_` 传给每个 pipeline 作为类别列元数据。LimiX 的 categorical encoding 在此数值矩阵上再次以 pipeline 层形式应用。
- `max_num_features` 若保留，改为每个成员在其最终特征视图上独立、确定性地采样；不允许使用当前共享的 group-index 数组，因为每个 pipeline 的输出维度可能不同。
- 每次 `fit` 持久化 `pipeline_specs_`、`pipeline_members_`、`pipeline_member_names_`、`pipeline_validation_audit_`、`pipeline_failed_members_`、`nnls_weights_`；预测不重新拟合任何步骤。
- 所有生成的默认成员均设置 `use_retrieval=False`；不引入 retrieval。

## 5. 测试

在 `tests/test_infer_package.py`（或拆分为新的 pipeline ensemble 测试文件）添加无需 checkpoint 的单元测试：

1. 默认 spec 数量及关键字段：分类严格为 32、回归严格为 8；回归前 4 条含 SVD，后 4 条 one-hot。
2. 每个 transformer 的 `fit/transform` 状态固定：列过滤、未知类别、one-hot 列宽、SVD 附加列、fingerprint、交互和 shuffle 在重复 transform 时确定且形状一致。
3. 分类标签置换的 inverse mapping 回到 canonical 类别概率顺序；每行概率仍归一化。
4. 单次验证仅调用一次 split，不调用 `StratifiedKFold` / `KFold`；分类拆分使用 stratify，回归不使用 stratify。
5. NNLS 分类与回归：权重非负、和为 1、成员掩码与权重长度/预测成员顺序一致；非有限成员被丢弃后剩余权重正确重归一化；退化时等权回退。
6. 通过 mock 的前向函数验证 classification/regression pipeline ensemble 的训练-验证-全量重拟合-预测状态流；有 checkpoint 的集成测试验证输出形状、有限值、概率归一化和 pickle 往返一致。

## 6. 验证命令

- `pytest -q tests/test_infer_package.py`
- 如可用 checkpoint：运行对应分类/回归推理测试和 save/load roundtrip。
- 对 `classifier.py`、`regressor.py` 做静态导入检查，确保删除的 KFold/group helper 不再被增强路径引用。
