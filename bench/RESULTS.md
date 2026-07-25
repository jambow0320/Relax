# Task 24 — Colocate 多模态性能:实测结果

环境:8×H200(143GB/卡),`relaxrl/relax:latest` 容器 `relax-task24`,worktree `task24-colocate-mm-perf`@039ce87。
脚本 `scripts/training/multimodal/run-qwen3-vl-4B-8xgpu-bench.sh`(仅日志后端改 tensorboard)。
模型 Qwen3-VL-4B-Instruct,数据 lmms-lab/multimodal-open-r1(单图/样本)。
配置:colocate,8 卡,TP=2,rollout-batch 64 × n-samples 8 = 512/步,max-response 8192,NUM_ROLLOUT=6(取稳态 steps 1–5)。

## 一、基线(2 次跑,steps 1–5 均值)

| 指标 | run1 | run2 | 均值 |
|---|---|---|---|
| step_time (s) | 116.0 | 118.6 | **117.3** |
| rollout / train_wait (s) | 77.2 | 79.1 | **78.1 (66%)** |
| train_time (s) | 38.8 | 39.5 | 39.2 (33%) |
| 切换 sleep+wake+update (s) | 11.3 | 10.8 | **11.1 (~10%)** |
| ├ sleep | 5.3 | 5.2 | offload Megatron |
| ├ wake_up | 1.4 | 1.4 | reload 20 NCCL 组 |
| ├ update_weights | 4.7 | 4.3 | 权重同步+onload KV |
| MFU actor_train | 0.104 | 0.106 | **~10.4%** |
| perf/rollout_time (s) | 61.3 | 64.3 | 62.8 |
| 峰值显存 / 卡 | — | 120.5 GB | **82%,无 OOM** |

可复现性好(run1 vs run2 差 <2.5%)。无 orphaned 进程(job 结束 GPU 干净归零)。

## 二、根因分析(定优化方向)

1. **rollout 是瓶颈(66%)**,且为**生成长尾**:SGLang 日志 `#running-req` 从 73 掉到 7、`total_lengths max=10040` vs mean≈1350;少数长序列拖尾、GPU 空转。
2. **不是显存/KV 受限**:SGLang `token usage 仅 1~6%` → `mem_fraction_static=0.8` 严重过剩(约 100GB/卡 KV 预留未用)。→ **调大 mem_fraction 无效**。
3. 图片为单张小几何图,`mm_encode_time` 仅 0.17s。

## 三、实验 A:开启多模态 processor pool(`--mm-processor-pool-size 16`)

现有 flag,零代码改动,agent 分析里的 #1 假设(图像预处理 GIL 争用卡在 rollout 关键路径)。

| 指标(steps 1–5 均值) | 基线 | procpool=16 | 变化 |
|---|---|---|---|
| image_processor_time / mean(每样本) | 3.12s | **1.98s** | **↓ 35%** |
| image_processor_time / max | 8.1s | **3.65s** | **↓ 55%** |
| **perf/rollout_time** | 62.8s | **64.2s** | **≈ 无变化** |
| step_time | 117s | 121s | 噪声内 |
| 峰值显存 | 120.5GB | 120.5GB | 无 OOM |

**结论(数据支持的否定):** processor pool 确实把图像预处理砍了 35~55%,**但 rollout_time / step_time 纹丝不动**。证明**图像预处理与生成重叠、不在关键路径**——单张小图的预处理早已藏在长尾生成之下。优化命中了目标指标,但目标指标不是瓶颈。这是有价值的影响评估:**在此工作负载下 processor pool 不改善端到端吞吐**(对图更多/更大的场景可能不同)。

## 四、下一步候选杠杆(各有取舍)

| 杠杆 | 预期 | 风险 | 类型 |
|---|---|---|---|
| B. 显存回收:降 `mem_fraction 0.8→0.4` | 释放 ~60GB/卡,吞吐不降(命中"显存回收"验收项) | 低 | 现有 flag |
| C. 减训练重计算 `recompute full→selective` | train 相位提速(仅 22GB 余量,可能 OOM) | 中 | 现有 flag |
| D. 切换开销:sleep 里 gc/host-cache 清理按 flag 跳过 | 省 ~1–3s/步 | 中(分布式路径) | 新 flag 门控代码 |
| E. 进程组复用:colocate 下不每步重建 20 个 NCCL 组 | 省 wake+部分 update(~2–4s/步) | 高(CLAUDE.md 硬约束,需 maintainer 认可) | 新 flag 门控代码 |
| F. 缓解生成长尾(partial rollout / 更大 batch 摊薄) | 直击 66% 大头 | 高(涉算法/训练语义) | 需对齐 |

## 五、Profiling(py-spy + SGLang profiler + tensorboard),定量根因

**方法**:重建容器加 `--cap-add SYS_PTRACE`;`--sglang-profile --sglang-profile-step-start 2 --sglang-profile-step-end 2` 抓 step2 生成 trace(8 引擎已落 `traces/profile-run/sglang_trace/rollout_2/`);py-spy 对 actor 主线程采样 61 次(~2 步)。

### 5.1 相位归因(py-spy,与 perf 指标交叉验证一致)
- **57% 主线程纯等待**(actor 已 offload、阻塞等 rollout)← rollout 相位
- **26% 在训练**(fwd/bwd/optimizer)
- **~13% 在切换**(update_weights 的通信+clear_memory、sleep、进程组重建)

### 5.2 rollout 长尾 = 核心瓶颈(响应长度分布)
| | min | median | mean | max | 截断率 |
|---|---|---|---|---|---|
| 每步典型 | ~14 | **~350** | ~1350 | **8192(封顶)** | 0~10% |

max 是 median 的 **23×**。SGLang `#running-req` 从 73 掉到 7 印证:中位序列早停后,引擎用极低并发单独解码少数 8192-token 长序列,GPU 空转。**同步 colocate 下训练必须等最长序列**,故 rollout 时间被长尾主导。

### 5.3 结论:无"安全 flag 银弹"
- rollout(66%)瓶颈是**固有的重尾生成**,唯一根治靠**算法/模式改动**(partial rollout / fully-async / 过采样早停),即杠杆 F —— 收益最大但需在 Issue 与 maintainer 对齐、涉训练语义。
- 可在不改算法下拿的是 **切换开销(~10-13%)**:进程组每步重建(E)、sleep 的 gc/host-cache 清理(D)—— 适合做 feature-flag 门控的 PR,但收益个位数 %。
- 训练(26-33%,MFU 10%)可试减重计算(C),受 22GB 余量限制、有 OOM 风险。
