# Sparse Model OPD 与分布式训练路线

## 0. 项目目标

构建并评估一套 dense teacher / sparse student 训练系统：

```text
候选模型 Benchmark
        -> OPD 最小训练闭环
        -> sparse student rollout / 训练
        -> TP 与多 GPU 扩展
        -> 质量和效率评测
```

当前不急着运行最大的模型。先建立可信的 benchmark 和单步训练闭环；每个阶段通过正确性门槛后，再扩大模型和 GPU 数量。

## 1. 当前已经验证的基础

- [x] 接入共享的 `catalyst-fleet1` B200 服务器。
- [x] 创建隔离的 `rockyeast-vortex` Conda 环境。
- [x] 安装官方 SGLang `0.5.12.post1` 和 Vortex plugin 依赖。
- [x] 验证 plugin discovery 和全部 9 个 hook target。
- [x] 通过 3 个 SGLang runtime contract 单元测试。
- [x] 在单张 B200 上通过 Vortex 使用 Qwen3-1.7B 真实生成 token。
- [x] 确认 CUDA kernel 针对 B200 `sm_100` 编译，并在结束后释放 GPU。
- [x] 在 AstraFlow 中加入 paired dense/Vortex serving benchmark runner。
- [x] 完成 Qwen3-1.7B 的 1K smoke、16K prefill-heavy 和 16K decode-heavy 对照。

当前证据覆盖单 GPU 推理和 Qwen3-1.7B benchmark 工具正确性。为绕过
SGLang `0.5.12.post1` 在 B200 上的 piecewise CUDA Graph 启动错误，dense 和
sparse 两侧都关闭了 piecewise CUDA Graph。它还不能证明 OPD 训练、TP、
多步稳定性、大模型支持或生产性能。

## 2. 阶段 A：建立可信的 Benchmark

### A1. 先用 Qwen3-1.7B 校验工具

在完全相同的设置下分别运行 dense SGLang 和 Vortex sparse attention：

- 相同模型 revision 和 dtype。
- 相同 prompt、随机种子、采样参数、上下文长度和输出长度。
- 相同 GPU 和进程生命周期。
- warmup 不计入正式测量。
- 每条命令都使用 `2h20m` 硬超时。

每次运行保存一个 JSON，至少记录：

- 模型、Git commit、环境版本、GPU 和完整配置。
- prompt token 数、生成 token 数和是否成功结束。
- 总延迟、TTFT、TPOT、generation throughput 和 request throughput。
- 峰值显存及稳定阶段的 KV cache 显存。
- dense/sparse 模式、`topk`、page size 和保留 page 比例。
- 生成结果或任务得分，用于质量检查。

### A2. 验证测量方法

- [x] Dense 运行成功并记录全部字段。
- [x] Sparse 运行成功并记录全部字段。
- [x] 正反运行顺序复核后，decode-heavy 结果方向一致。
- [x] Sparse 日志生成 Vortex cache/indexer compiled function，排除 dense fallback。
- [x] 每次运行后本任务进程退出，GPU 显存恢复到基线。
- [x] 失败任务抛出错误并保存 server/benchmark 日志，不会静默漏掉。

### A2 当前结果（2026-07-23，单张 B200）

Runner：
`/home/rockyeast/astraflow/src/scripts/benchmark_vortex_serving.py`

远端 artifact：
`/home/zhuominc/rockyeast-work/benchmark-results/`

| Workload | 结果 | 结论 |
|---|---:|---|
| 1K input / 64 output / 16 requests / c=4 | sparse throughput = dense 的 93.1% | 工具 smoke 通过；上下文太短，不用于性能结论 |
| 15,360 input / 128 output / 64 requests / c=8 | sparse throughput = dense 的 90.5% | prefill-heavy；Vortex 额外准备成本未被短 decode 摊平 |
| 12,288 input / 2,048 output / 16 requests / c=4 | sparse throughput = dense 的 99.7% | decode-heavy 基本持平；TPOT 约改善 0.9% |
| 同上，反向顺序 sparse -> dense | sparse throughput = dense 的 99.3% | 排除明显顺序偏差；1.7B 上无显著加速 |

三组运行均无请求错误，短确定性质量探针输出完全一致。该探针只证明启动
和生成链路一致，不代表长上下文任务质量。Qwen3-1.7B 在 B200 上主要用于
验证工具；它太小，不能据此否定更大模型上的 sparse attention 收益。

### A3. 逐级扩大模型

只有 A2 通过后才继续：

1. Qwen3-1.7B：只用于验证 benchmark 工具，不作为最终研究结论。
2. Qwen3.5-9B：第一组有实际参考意义的单 GPU 成本数据。
3. Qwen3-32B：候选 dense teacher 和大模型 rollout 成本。
4. 可选的 30B/35B MoE：先确认 Vortex 是否支持该架构，并明确为什么要测，再运行。

至少准备两类 workload：

- 推理题：测 rollout 时间和任务质量。
- 长上下文：测 sparse attention 加速和 KV memory 行为。

### 阶段 A 退出条件

- [ ] 至少完成 1.7B 和 9B 的可复现 dense/sparse 对照表。
- [ ] 完成一个更大模型的测量，或明确记录阻塞原因。
- [ ] 根据显存、rollout 时间、质量和 Vortex 兼容性，选出有依据的 teacher/student 组合。

## 3. 阶段 B：打通最小 OPD 训练闭环

先做最小端到端 iteration，不要一开始就接分布式框架：

```text
prompt batch
  -> 当前 student 生成 trajectories
  -> teacher 给同一批 student token 打分
  -> student_logprob - teacher_logprob = 逐 token reverse KL
  -> negative reverse KL 作为 token advantage
  -> backward + optimizer step 更新 student
  -> 下一轮使用更新后的 student 重新采样
```

尽量从单 GPU 和最小模型配置开始。

正确性检查：

- [ ] prompt token 不参与 distillation loss。
- [ ] teacher 和 student 对齐到完全相同的 response token 位置。
- [ ] teacher 参数不产生梯度。
- [ ] student 梯度和 loss 都是有限值。
- [ ] optimizer step 后至少一个 student 参数发生变化。
- [ ] checkpoint 保存和重新加载后仍是更新后的模型。
- [ ] 先运行 1 step，再运行 5-10 steps，不能出现显存泄漏或死锁。

### 阶段 B 退出条件

- [ ] 保存可检查的 trajectory、两组 log-prob、token mask、reverse-KL advantage、loss 和参数变化证据。
- [ ] 有一条可重复执行的短程多步 OPD 命令。

## 4. 阶段 C：定义并训练 Sparse Student

实现前必须先和学长确认一个边界：

```text
Vortex sparsity 只用于 student rollout 推理，
还是 backward / 训练阶段的 attention 也必须稀疏？
```

这是两种不同的工程范围。现在 Vortex 已证明 serving/rollout 路径可用，但它不会自动提供可反向传播的 sparse training kernel。

确认边界后再完成：

- Dense teacher 给 student 生成的 trajectory 打分。
- Student rollout 使用选定的 Vortex sparse-attention 策略。
- 训练阶段使用约定好的 dense 或 sparse backward 实现。
- 更新后的权重同步回 rollout engine。
- Dense student 和 sparse student 使用相同 prompt 与训练预算。

### 阶段 C 退出条件

- [ ] 下一轮 rollout 确实读取了更新后的 student 权重。
- [ ] 权重更新后 sparse rollout 仍然生效。
- [ ] 多步训练没有质量崩溃或非法 KL。
- [ ] 能在相同训练预算下比较 dense student 和 sparse student。

## 5. 阶段 D：TP 与多 GPU

单 GPU 训练闭环正确后才扩展：

1. TP=1 基线。
2. TP=2 正确性测试。
3. 只有模型规模或吞吐确实需要时才运行 TP=4。
4. 单个模型的 TP 跑通后，再考虑 teacher/student 分开放置。

验证内容：

- [ ] 各 TP size 能加载同一 checkpoint。
- [ ] 输出质量和 log-prob 与 TP=1 在合理数值误差内一致。
- [ ] startup、generation、backward、shutdown 都没有 rank 卡死。
- [ ] 记录每个 rank 的显存和通信时间。
- [ ] 多 GPU 带来的有效吞吐提升足以抵消 TP 通信开销。

在学长提供合适资源前，multi-node 不进入当前范围。

## 6. 阶段 E：最终评测

最终报告必须分开三类指标，不能只写一个综合 speedup。

### 质量

- 任务 accuracy 或 pass rate。
- Student 相对 dense teacher 的质量保留率。
- Dense student 与 sparse student 的质量差异。
- KL 趋势和典型失败样例。

### Serving / Rollout 效率

- TTFT、TPOT、tokens/s、requests/s。
- 峰值显存和 KV cache 显存。
- 保留 KV page 比例与上下文长度。

### 训练效率

- 达到相同质量所需的 GPU-hours 或总训练时间。
- 分开记录 teacher scoring、student rollout 和 backward 时间。
- TP scaling efficiency 和通信开销。

`97% teacher quality`、`1.5x throughput`、`3x less training compute` 等数字在产生对应实验 artifact 之前都只是目标，不能当作已完成结果。

## 7. 共享 B200 使用规则

- 每次任务前立即运行 `nvidia-smi`。
- 即使进程属于同一个 `zhuominc` Unix 账号，也不能终止其他人的任务。
- 使用 `CUDA_VISIBLE_DEVICES=<空闲 GPU 编号>` 限制可见 GPU。
- 每个 GPU 命令都包上：

```bash
timeout --signal=TERM --kill-after=30s 2h20m <command>
```

- 每次运行后检查显存是否释放、是否有残留进程。
- 复用已有 Hugging Face cache，不重复下载大 checkpoint。
- 个人代码和结果放在 `~/rockyeast-work/`。

## 8. 下一次 GPU 运行

```bash
ssh catalyst-b200
cd ~/rockyeast-work/astraflow-opd
conda activate rockyeast-opd
git status
nvidia-smi
```

阶段 A 的 runner 和 Qwen3-1.7B 工具校验已经完成。下一步先对缓存中的
Qwen3.5-9B 做最小 dense/sparse token smoke，确认模型架构与 Vortex
兼容；通过后再运行同一套 16K 对照。若 Qwen3.5 架构不支持，记录阻塞点并
改测标准 GQA 架构，不要为了跑 benchmark 临时改 Vortex 核心逻辑。

## 9. 最终交付物

- 带确定性配置和硬超时的 benchmark runner。
- 每次运行对应的机器可读 JSON artifact。
- Dense/sparse 对照表和图。
- 可检查逐 token 数据的最小 OPD 训练脚本。
- Sparse student 多步训练结果。
- TP scaling 表。
- 明确区分成功、失败和后续工作的最终报告。
