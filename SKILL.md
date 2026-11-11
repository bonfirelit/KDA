---
name: kernel-opt
argument-hint: "<kernel 绝对路径或名字> [生产形状 / 验收关注点等补充说明]"
description: >
  Bootstrap a kernel optimization sandbox from an upstream kernel (verbatim
  baseline, frozen test/bench/race harness, TASK.md + ITERS.md) and run the
  campaign in fresh subagent rounds. Each round one new agent (model rotated)
  reproduces the current tier's accepted state and works candidates; the main
  session counts per-tier stale rounds and stops when the lowest applicable
  tier is exhausted. Follows the abstraction ladder: start at the origin's own
  tier (DSL / native CUDA-HIP / PTX-asm) and descend only after bottleneck
  attribution shows the remaining gap is due to the current tier's
  expressiveness or codegen. Platform-agnostic: probe platform, arch,
  compiler, profiler, and device selection at bootstrap. Use when asked to
  "optimize kernel X", "start/continue a kernel sandbox/campaign", or mentions
  kernel-opt. NOT for one-off tuning or plain kernel Q&A.
---

# kernel-opt — 沙箱 + 多 agent 轮换制优化

**平台**：任意 GPU/NPU。平台、arch、编译器、profiler、选卡与钉卡方式都由
bootstrap 探测成机器档案（§1.0），不硬编码。前提：工具链可用，profiler 能给出
可比数字。

**优化阶梯**：

- t1 DSL：Triton、TileLang 等。
- t2 原生：CUDA / HIP（含 C++ 模板库写法）。
- t3 低层：PTX / 内联汇编 / arch intrinsic（NVIDIA PTX、SASS；AMD GCN asm）。

起始层 = origin 的实现层：DSL → t1；CUDA/HIP → t2（不绕 DSL）；PTX/asm → t3。
当前层到顶且瓶颈归因属于层级性上限时，把该层 accepted 实现翻译到下一层继续
优化；最低适用层到顶才收尾。翻译保留已验收的算法、分块、流水、融合决策。

**附件**（用到才读）：

- `templates/TASK.md` — 建沙箱时读。
- `templates/ITERS.md` — 建沙箱与收尾时读。

## 0. 停止与换层（唯一权威规则）

只看轮裁决。stale 计数按层独立。每轮结束后归入两类之一：

- **stale**：本轮所有 candidate 提升都在噪声内 / 被否决 / 作废，且 agent 明确
  报告「没有进一步空间」并附理由（结构空间已扫、触顶证据、被否决方向）。
- **progress**：有 candidate accepted，或全败但列出「尚未试」的具体假设。
  中途死亡 / 环境被占不算 stale，不计数。

当前层连续 3 个 stale → 该层到顶；任一 candidate accepted 则当前层计数归零。
「没空间」必须有结构空间扫描支撑（多个方向各写最小探针、用数字否决），否则
该轮不算 stale，打回补扫描。

到顶后：

1. **主会话做瓶颈归因**。只有 profiler / 反汇编证据表明剩余差距来自当前层
   表达力 / codegen（寄存器与占用率控制、spill、异步拷贝/流水/warp
   specialization、特定指令如 `cp.async.bulk` / `wgmma` / `ldmatrix` / cache
   policy、指令调度）才允许换层。证据写进 TASK.md 层级计划，转移时写进 ITERS。
   无归因证据不换层。
2. 归因属于物理 / 算法上限（实测带宽、roofline、算法结构）→ 不换层，收尾（§3）。
3. 层级性上限且有更低可用层 → 层级转移（§1.2），新层计数归零。
4. 已是最低适用层 → 收尾（§3）。

**触顶谓词命中**：主会话先验非空洞性——TASK.md Profiling 节必须登记过对照
实验（空 kernel / memset 不触发），busy 率计数器（GRBM_*、SM/队列 busy 等）
永不作证据。非空洞则当前层立即到顶，按上面 1-4 处理；空洞或谓词未声明则本轮
不算，下一轮点名要求补对照实验或撤回结论。

## 1. Bootstrap（只做一次）

### 1.-1 入口

```
/kernel-opt <kernel 的绝对路径或名字> [生产形状等补充说明]
```

解析成三要素：

- **origin**：路径直接用；只是名字就先定位到文件，把绝对路径和**实现层**
  （t1/t2/t3，决定起始层）复述确认。
- **brief**：kernel 是什么、为什么优化、生产形状、验收关注点。生产形状缺省时
  从调用点 / 模型结构推导候选值，列给用户确认。
- **sandbox**：`<名字>-opt`，放在用户指定工作区根下；路径复述确认。

### 1.0 机器档案

在目标机逐项实测，写成一段文字，注入 build 命令、TASK.md 命令表和每轮 prompt。
不要写别台机器的值。

| 项 | 怎么探 |
|---|---|
| 平台 + arch + warp/wave size | 确认后端（CUDA / ROCm / 其它）。arch：AMD `rocminfo \| grep -i gfx`；NVIDIA `nvidia-smi --query-gpu=name,compute_cap` 或 `deviceQuery`。warp/wave size：CUDA 32；AMD RDNA wave32、CDNA wave64 |
| 加速器数/显存/占用 | `nvidia-smi` 或 `rocm-smi`（数量、每张显存、util%）→ 选空载卡钉为校准卡，记卡号 |
| 访问纪律 | 读项目 / 全局 CLAUDE.md；`groups` 看权限组；判断是否需包裹（`sg`、容器、直接跑）。没有包裹就直跑 |
| 钉卡变量 | AMD `ROCR_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES`；NVIDIA `CUDA_VISIBLE_DEVICES`；Triton 跟随底层后端 |
| 工具链 | `which` + 版本命令（CUDA: `nvcc ncu nsys nvidia-smi`；ROCm: `hipcc rocprofv3 rocminfo rocm-smi`）；缺失即停手报告 |
| 层级能力 | t1：origin venv 里 `python -c "import triton; print(triton.__version__)"`（TileLang 同理）；t2：`nvcc`/`hipcc`；t3：PTX 内联汇编（NVIDIA）或 GCN/LLVM asm（AMD）。记录可用层集合与最低适用层；不可用层写 N/A + 证据 |
| python/venv | 用 origin 项目 venv 的 python 全路径；**PATH 必须带 venv/bin**，否则 JIT/后端静默回退 |
| 辅助 skill | 发现范围：当前会话可用 skill 列表 + `~/.claude/skills` + 项目 `.claude/skills` + 插件 skill。分两类记录：**知识库**（kernel wiki 等，立假设时查经验）与 **profiler 报告分析**（ncu/rocprof report 解析等，取证时用）。每项写「skill 名 + 用途 + 调用方式」；无则记「无」，不编造 |
| profiler 计数器 | 先特征化再信任：空 kernel / memset / 单块延迟链对照，登记可信与空洞计数器 |

两条通用警示：

- **busy 率类计数器可能是空洞证据**：对空 kernel / memset 也读满（队列级 busy
  而非计算单元饱和），不能当占用率或触顶证据。任何计数器用作证据前先做空
  kernel + memset 对照；空洞清单随平台/arch 变。
- **warp/wave size**：CUDA 32；AMD RDNA 32、CDNA 64；影响编译宏与
  `__shfl`/协作组假设。以 `rocminfo` 等实测为准。

### 1.1 建沙箱

目录结构（`(冻结)` 为冻结清单成员，其余为可改载体；`<ext>` 为 `.cu`/`.hip`
等）：

**原生编译（origin 是 CUDA/HIP 时作为 t2 起始层）**

```
<sandbox>/
  build.sh              # <编译器> <arch flag> -std=c++17 -O3 -I include -I src
  test.<ext>  -> ./test   # 正确性（冻结）
  bench.<ext> -> ./bench  # 性能（冻结）；iter-0 后拷 bench_base（.gitignore）
  race.<ext>  -> ./race   # 大 grid 确定性探针（冻结）
  TASK.md  ITERS.md  .gitignore
  src/
    <kernel>.<ext>      # ← 载体
    harness.hpp         # (冻结) host 驱动
    ref.hpp             # (冻结) 独立参考实现
  include/
    architecture_config.h  # (冻结)
```

编译器与 arch flag 按机器档案（如 `hipcc --offload-arch=<arch>`、
`nvcc -arch=sm_<xx>`）。

**Triton（origin 是 DSL 时作为 t1 起始层）**

```
<sandbox>/
  kernel_baseline.py    # 上游逐字拷贝，仅 import 改写（冻结）
  kernel_v1.py …        # ← 载体：每 candidate 一个新文件
  test.py  bench.py     # importlib 按模块名加载（冻结）
  TASK.md  ITERS.md  .gitignore
```

Triton 无 build（JIT）；测试/基准用上游 venv 的 python 全路径，后端由环境决定。

**层级叠加**：只建起始层。转移时在同一沙箱加子目录（如 `tier2_native/`、
`tier3_lowlevel/`），上一层 accepted 实现保持冻结、可编译、可运行；不原地
改写或删除上一层文件。

按 `templates/TASK.md` 与 `templates/ITERS.md` 创建：

1. **逐字拷贝基线**：kernel 源从 origin 原样拷入；机械适配仅限 include 扁平化、
   import 改写，逐条记录（commit message + TASK.md 溯源节）。
2. **flavor 判定**：原生编译（build.sh + 档案的编译器与 arch flag）或 Triton
   （kernel_baseline.py 冻结，每 candidate 一个 kernel_vN.py；arch 由 JIT 从
   设备推导）。
3. **写冻结 harness**：test（数值层 + 形状覆盖 + golden hash + 大 grid 确定性
   探针：≥8 reps、毒化输出、位级比对，至少一个 ≥8×8 块的大 grid）、bench
   （生产形状、median-of-medians、交错 A/B、json 落盘）、race 探针、TASK.md、
   ITERS.md。冻结清单 = 载体之外的一切；给出
   `git diff --exit-code -- <清单>` 校验命令。
4. **基线全绿**：build、test、race 在空载卡上跑通；sandbox-vs-origin 文件级
   diff 只含已记录适配。
5. **原生 flavor**：iter-0 的 `./bench` 拷成 `bench_base`（.gitignore），后续
   重建不动。
6. `git init` + `iter-0: baseline` commit（溯源 + 适配清单）。
7. **校准**：基线 bench 3 轮（原生 `./bench_base` ×3；Triton
   `bench.py kernel_baseline` ×3，PATH 必带 venv/bin），主指标中位数，3 轮
   spread → noisePct 写 TASK.md；完整基线表写 ITERS iter-0；单独
   `calibration` commit。校准前先特征化 profiler 计数器。
8. **突变测试**：`/tmp` scratch 上对载体施加 4 类突变（sync / boundary /
   coverage / numeric，每次一处），重建跑全套测试，每类必须被抓住。有存活 →
   只强化 harness（独立 commit），重试。scratch 用 `rm -rf <dst> && cp -r`；
   sync 类独占静置卡。
9. 把 noisePct、冻结清单、验收规则交用户 review，再开始轮换。已有沙箱直接从
   §2 进入。

### 1.2 层级转移（主会话执行）

**触发**：当前层到顶（§0），瓶颈归因为层级性上限，且还有更低可用层。转移不是
candidate；候选 agent 不碰 harness、不做跨层翻译。

1. **冻结上一层**：accepted 实现及 build 脚本加入冻结清单，保持可编译/运行；
   记录 commit 与最终数字。
2. **翻译**：以该层 accepted 实现为蓝本（不是 origin 原版），把算法、分块、
   流水、融合、布局决策翻译到下一层语言；记录只属于上层的约束。
3. **搭新层脚手架 + 改 harness**：新建层目录与 build/绑定，把新实现注册进
   test/bench/race 的 impl 调度；harness 变更独立 commit
   `harness: tier-N support`，同步冻结清单与命令表；确认上一层仍可跑。
4. **跨层 A/B（同 session）**：上一层 accepted 与新层基线同 session 交错（能同
   进程就同进程，否则顺序跑并记录限制）。新层初版可落后，记录差距；该层目标
   最终超过上一层；到顶仍落后则整体 best 是上一层。
5. **新层全绿**：test 全 PASS、race 全 OK；翻译改变累加顺序时按 RECORD ME
   重录 golden，且先过数值层。
6. **重算新层 noisePct**：3 轮 A/A，不同层不共用。
7. **更新 TASK.md / ITERS.md**：当前层、命令表、冻结清单、层级计划；ITERS
   追加 tier-N 条目并写下穿归因证据；commit
   `tier-N: baseline from tier-M accepted`。
8. 新层 stale 计数归零，回到 §2。

翻译可交给专门翻译轮 agent（只改新层载体），但 harness/绑定/命令表由主会话
先搭好；该轮不计 stale。

## 2. 轮换制

### 启动一轮

用 Agent 开 `subagent_type: "general-purpose"` 的后台 agent，模型在 `opus` 与
`sonnet` 之间轮换（传 `model:`）。一轮一个，不并行（A/B 钉同一卡，冻结校验
要求工作树串行）。

prompt 模板（不要把 TASK.md 内容贴进去）：

```
你是沙箱 <绝对路径> 的本轮优化者（轮 R<k>，当前层 t<N>）。

只在当前层工作：不跨层翻译、不改 harness、不动上一层文件。判断到顶时按裁决
格式汇报，由主会话决定换层或收尾。

先读：
1. TASK.md，按首节复现当前层最近 accepted 状态：git status → 冻结校验 →
   test → 一次 bench 对 ITERS.md 当前层数字。
2. ITERS.md 全部条目，避免重复已否决方向。

按 TASK.md 工作流做 1-3 个 candidate：立假设 → 查知识库（档案有知识库 skill
则用 Skill 工具，查不到记「无」）→ 只改当前层载体 → 冻结校验 → test → race →
同 session 交错 A/B（before 为同 session 重测基线）→ 取证（档案有 profiler
报告分析 skill 则用它解析 report）→ ITERS 追加 wip → commit 或 revert。验收
规则与 noisePct 见 TASK.md。

硬规则：
- test 挂 = candidate 作废，bench 不跑，revert 后继续下一个假设或结束本轮。
- 一 candidate 一 commit/revert，工作树保持干净或已提交。
- 命令用 TASK.md 命令表逐字版本；命令本身坏了 → 停手报 envBlocked，harness
  只能由主会话修。
- 复现失败：先查 drift，不优化。
- 环境被占：停手报 envBlocked。
- profiling 只用已特征化计数器，包裹命令照抄命令表。

汇报必须含：
(a) 每个 candidate 的结局与一句话理由；
(b) 「有进展/有具体假设未试」或「没有进一步空间」（附结构空间扫描、触顶对照、
    已否决方向）；
(c) git log 最近 3 条与 git status。
设备纪律与机器档案：<逐字粘贴 §1.0 档案，含包裹方式与校准卡号>。
```

一轮 1-3 个 candidate 由 agent 自裁量。

### 一轮结束后主会话做四件事

1. **验收**：`git -C <sandbox> status` 干净、log 与汇报一致、ITERS 新条目与
   bench json 对得上（抽一个数复核）。
2. **裁决归类**：progress（当前层归零）或 stale（+1；无证据的「没空间」打回）。
3. **决定**：当前层到顶（stale=3 或非空洞触顶）→ 有更低可用层走 §1.2，否则
   §3；未到顶则换模型开下一轮，prompt 附上轮结论。
4. 转述用户：本轮结局、累计收益、当前层 stale、下一层计划。

### 中途死亡 / 环境故障

agent 死亡、envBlocked 或长时间无进展（transcript 末尾 API Error、文件数分钟
未更新、工作树干净）：

1. 查设备占用与 `git status`。
2. 树脏则 `git checkout -- . && git clean -fd`（Triton 另删多余 kernel_vN.py）。
3. 换模型重启；API 死的别等同模型恢复。
4. 不计 stale。

## 3. 收尾（最低适用层连续 3 stale 后）

主会话做：

1. ITERS 末尾追加终止小结：终止条件、各层旅程（各层是否进入、最终数字、收益、
   跳过原因）、最终采纳层、终态 vs iter-0、总收益、证据指针、剩余空间。最终
   结果 = 各层 accepted 中主指标最优者，写明所属层。
2. `termination` commit。
3. 转述用户：accepted 迭代、各层收益与最终采纳层、总收益、剩余空间。合回原
   仓库由用户另行发起。
