# Qwen3-30B-A3B Prefill Router Trace

## 当前交付状态（2026-10-17）

**尚未取得真实 Router Trace，阶段一未通过，未开始扩展采集。** 当前仓库提供采集工具、环境实测和阻塞记录，不能被引用为已完成的数据集。模型未加载、未做 Forward、未发生 OOM；没有伪造路由、统计或运行结果。`traces/` 没有实际数据文件。采集代码尚未完成真实模型端到端验证。

主要阻塞是官方权重大文件下载速度。官方 16 个 safetensors 分片共 **61,066,575,648 字节**。三个并行下载约 285 秒，只写入 75,497,472 字节（约 0.265 MB/s），外推约 **64 小时**，不含校验与运行。单独 1 MiB Range 探测用时 25.18 秒。直连失败，代理访问元数据正常。探测下载已停止，可续传分片保留在 `/data/qwen-router-cache/model/`，没有提交。详见 `reports/availability.json`。

## 资源实测与计算方式

- 两张 NVIDIA A100-PCIE-40GB：初始占用约 4.0GB、31.4GB；可用约 36GB、9GB。没有停止他人进程。
- NVIDIA Driver 570.133.20；驱动支持 CUDA 12.8；PyTorch 实际 CUDA runtime 12.6。
- PyTorch 2.6.0+cu126；独立环境固定 Transformers 4.51.3、Accelerate 1.6.0、tokenizers 0.21.1、huggingface-hub 0.30.2、NumPy 2.2.6。
- 宿主机内存 628GiB，但 **容器 cgroup 限额只有 16GiB**。CPU 权重不能按宿主机内存预算加载。
- 根盘约 15GB 可用；`/data` 约 31TB 可用。权重与 offload 都放在 `/data`。
- 计划采用 **BF16 + 双 GPU + 磁盘 offload**，CPU 常驻权重预算为 0；GPU0 预留 6GiB，GPU1 预留 3GiB。Accelerate 将 offload 权重搬到 GPU 上计算，不量化。该布局尚未实际加载验证。
- 模型 Forward 耗时和 GPU 显存峰值目前是 **未测量**，不是 0。成功运行会写入每个 Batch 的 JSON。`monitor.py` 另记录包括其他进程占用在内的设备总显存，不能与 PyTorch 本进程峰值混淆。

## 固定来源

- 模型：[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39)。模型与 Tokenizer revision 均为 `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。
- 官方配置：48 个 MoE Layer，每层 128 个 Expert，每个 Token 选 8 个；配置原文在模型缓存中，API 快照在 `reports/`。
- 模型实现：[Transformers v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。`reports/official_moe_forward.txt` 保存本地安装版本的官方 MoE forward，保留原始版权（Apache-2.0）。
- 输入：[THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672)，revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，`narrativeqa`。本次只准备这一来源，不声称覆盖 GPQA/Codeforces；其可访问 API 元数据也已记录。

## 准备与运行

以下命令从本仓库根目录执行。当前 `.venv` 已使用服务器的 PyTorch 建立；迁移到其他机器时，先安装匹配 CUDA 的 PyTorch 2.6.0，再安装固定依赖。服务器原有 Transformers 5.8.0 与 tokenizers 0.23.1 不兼容，没有修改原环境。

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 完整准备：需要下载约 61GB 权重；当前网络预计耗时数天。
# 下载后核对 HF API 中 LFS SHA256，生成 model_hashes.json。
.venv/bin/python scripts/prepare.py

# 只准备 Tokenizer 和真实文本输入，不启动模型：
.venv/bin/python scripts/download_tokenizer.py
.venv/bin/python scripts/fetch_longbench_subset.py
.venv/bin/python scripts/prepare.py --inputs-only

# 一次只启动一个大型模型。第一次运行建议仅阶段一。
.venv/bin/python scripts/monitor.py .venv/bin/python scripts/collect.py --minimal-only

# 阶段一通过后，扩大规模；仍只有一个实例。
# 此命令会再次执行最小验证与重复验证，再逐个执行扩展 Batch。
.venv/bin/python scripts/monitor.py .venv/bin/python scripts/collect.py

.venv/bin/python scripts/validate.py traces/*.npz
.venv/bin/python scripts/stats.py
.venv/bin/python scripts/checksum.py
sha256sum -c SHA256SUMS
```

已有同 revision 的权重时，可以将已校验文件放入缓存目录，或修改 `common.py` 的 CACHE 后重新准备。`prepare.py` 会检查已有大文件的官方 LFS SHA256，不能只凭文件名判断相同模型。不要将未下载完整的 `.part` 改名为 safetensors。断点续传验证 HTTP Content-Range；不支持 Range 时从头写入临时文件；哈希不符立即停止。

## 输入定义与 Batch

Prompt 为 `Read the following story and answer the question.\n\n` + 原始 `context` + `\n\nQuestion: ` + 原始 `input` + `\nAnswer:`。不使用 chat template，`add_special_tokens=False`。先对完整 Prompt 分词，再截取 token 前缀。

**所有计划请求都是明确标记的真实文本截断片段，不是完整原始 LongBench 请求**；较长原文的提问可能不在片段里。没有随机 Token、拼接伪装或循环填充。`data/requests.json` 保存原始样本 ID、行号、原 Prompt SHA256、原始 token 长度、截取范围、实际 input_ids 和片段文本。原始完整文本留在缓存，不进入 Git。

计划：最小验证 192 Tokens；单请求分别为 256、512、1024、2048、4096、8192 Tokens；另外 256、1024、4096 Token 的双请求 Batch，长度比例 1:3，以覆盖右侧 Padding。`data/batches.json` 记录请求数量、各自长度和有效 Token 总数。位置为原 Prompt 的 0-based Token 索引；当前均从位置 0 开始。

## 观察位置与验证

`observer.py` 使用 CPython `sys.settrace`，只读取官方 `Qwen3MoeSparseMoeBlock.forward` 的局部变量，不替换 forward，不改写 topk、Attention、Expert 或张量。关闭逐行事件，只在官方 MoE forward 返回时读取本次真实计算使用的 `selected_experts`、BF16 cast 后的权重。源码固定检查包含官方 topk 和基于 selected_experts 的 one-hot dispatch。

每个 Expert 的 forward pre-hook 统计实际接收的 Token 数，逐层核对实际调用次数与 Top-k 直方图（此核对包含 Padding；导出时才排除 Padding）。阶段一在每层 Expert 0、32、64、96 非空时还核对其输入张量与官方 frame 的 hidden_states/top_x/idx 完全相同，并核对这些派发位置对应的 Expert ID；同时从真实 router_logits 复核官方 softmax/topk。复核结果不会替代实际观测值。

每个 Batch 调用一次 `model(...)`，`use_cache=False`，不调用 `generate()`，不 Decode，不训练。`logits_to_keep=1` 只限制 LM head 输出到最后一个位置，所有输入 Token 仍经过全部 Attention/MoE 层。Attention 固定为官方 SDPA 实现，不重新实现计算。禁止调试器与 trace observer 同时运行。

验证器检查 `(有效 Tokens, 48, 8)`、0～127 范围、每行 8 个不同 ID、每个请求的逐位置覆盖、input_ids、layer_ids、request_ids 以及权重归一化。阶段一将同一输入同一 Batch 连续运行两次，逐槽精确比较。设置 deterministic algorithms、固定 CUBLAS 工作区、关闭 TF32。若非确定算子报错、路由不一致、OOM 或其他错误，记录并停止，不自动重试或扩大规模。跨设备、内核或 batch shape 的 BF16 路由不承诺逐位一致。

**以上是代码实现的检查，尚无真实模型运行通过的结果。** 静态语法、依赖导入与 CUDA 可见性验证不能替代阶段一。

## 输出格式

每个 Batch：`traces/<batch_id>.npz` 与同名 JSON。NPZ 不使用 pickle：

- `expert_ids`: uint8，`[有效 Token, 48, 8]`，保留官方 Top-k 顺序。
- `router_weights`: float32，保存官方 BF16 cast 后权重转 float32 的值。
- `request_index`、`request_ids`、`token_position`、`token_id`、`layer_ids`：完整关联关系。
- JSON：Batch、请求长度、revision、耗时、GPU allocated/reserved 峰值、Hook 核验结果。

导出完整 CSV 或单层：

```bash
.venv/bin/python scripts/export_csv.py traces/single_256.npz all_layers.csv
.venv/bin/python scripts/export_csv.py traces/single_256.npz layer_17.csv --layer 17
```

单层也可直接取 `np.load(path)['expert_ids'][:, 17, :]`，并同时保留请求与位置数组。每个 Batch 独立文件，request_id 在计划中全局唯一；重复验证有意复用输入 request_id，用文件/Batch ID 区分。

`stats.py` 逐 Batch 逐层输出各 Expert 选择次数、max、mean、max/mean、Jain fairness：`(sum(load)^2)/(128*sum(load^2))`。图包括每层负载热图和不同有效 Token 规模的公平度曲线。重复验证文件不计入统计。没有实际 Trace 时拒绝生成图。

`SHA256SUMS` 覆盖当前交付代码、说明、元数据和实际存在的数据文件。模型、Tokenizer 及数据源的哈希单独在 reports 中记录；模型权重与缓存不提交。尚未完成的下载没有完整文件哈希。

## Git

本任务专用仓库为 https://github.com/ponedo/qwen-load ，独立分支为 `qwen-router-trace`。仅提交代码、说明、环境报告和适当大小的真实文本输入；不提交模型权重、缓存或未完成下载。真实 Router Trace 尚未采集，等待官方权重可用后继续阶段一。
