---
name: kernel-opt
argument-hint: "<kernel 绝对路径或名字> [生产形状 / 验收关注点等补充说明]"
description: >
  Bootstrap a kernel optimization sandbox from an upstream kernel (verbatim
  baseline copy, frozen test/bench/race harness, TASK.md + ITERS.md), then run
  the campaign by spawning fresh subagent rounds — each round one new agent
  (optionally a different model) that reads TASK.md/ITERS.md, reproduces the
  latest accepted state, and works candidates; the main session counts rounds
  and stops after 3 consecutive "no headroom" verdicts. Use when the user asks
  to "optimize kernel X", "start a kernel sandbox for X", "continue the kernel
  campaign", or mentions kernel-opt. AMD ROCm platforms only (any gfx arch —
  RDNA3/4, CDNA3/4 — machine profile is probed at bootstrap, not hardcoded);
  HIP (native hipcc) and Triton-on-ROCm flavors. NOT for NVIDIA/CUDA work,
  one-off tuning, or plain kernel Q&A.
---

# kernel-opt — 沙箱 + 多 agent 轮换制优化

**平台范围：AMD ROCm only**（HIP native 或 Triton-on-ROCm），但**不限定 arch**——
RDNA3/4（gfx11xx/gfx12xx）、CDNA3/4（gfx94x/gfx95x）都行。机器与 arch 特定信息
不由本 skill 硬编码，而是 bootstrap 第 0 步探测成一份**机器档案**注入后续所有
prompt（见 §1.0）。

**按需查阅的附件**（都在本 skill 目录下，用到才读，不要预先全读）：

- `templates/TASK.md` — 沙箱 TASK.md 的填写模板（章节顺序固定，含正确性门禁、
  性能判定、工作流、终止条件、命令表、Profiling 各节）。**建沙箱时读它**。
- `templates/ITERS.md` — ITERS.md 模板（条目模板 + iter-0 校准条目 + 终止小结
  模板）。**建沙箱与收尾时读它**。

这两份模板是方法论的载体：10 条核心不变量（基线逐字拷贝、冻结清单、test 挂即
作废、同 session 交错 A/B、分层正确性门禁、噪声阈值来自校准、profiler 先特征化、
追加式日志、一迭代一 commit/revert、先查知识库再动手）都编码在模板章节里，
照着模板填就是照着纪律做。

## 0. 什么时候停（唯一权威规则）

停不停只看**轮裁决**，不看单个 candidate 成败。每个 agent 轮结束后，主会话把该轮
归入两类之一：

- **stale**：本轮所有 candidate 提升都在噪声内 / 被否决 / 作废，**且** agent 在
  最终汇报里明确给出「没有进一步空间」判断（附理由：结构空间已扫、触顶证据、
  被否决方向清单）。
- **progress**：有 candidate 被 accepted，或 agent 虽然全败但明确列出了
  「还没来得及试」的具体假设清单。**agent 中途死亡/环境被占（GPU 忙）不算
  stale**——修好环境直接重启一轮，不计数。

**连续 3 个 stale 轮 → 停**。accepted 任何一个 candidate 即重置计数为 0。
agent 说「没空间」不能光凭嘴说：主会话检查它至少做过一轮**结构空间扫描**
（多个候选方向各写最小探针测量、用数字否决，而不是空想后直接放弃），
否则该轮不计为 stale，打回让它把扫描做完。

**另一条立即停的路径：触顶谓词命中**。某轮 agent 报告命中 TASK.md 里声明的
触顶谓词（如「decode weight GB/s ≥ 0.9×实测 HBM 有效带宽」）时，主会话先查
**非空洞性**——该谓词必须已在 TASK.md Profiling 节登记过对照实验（空 kernel /
memset 不触发它），busy 率类计数器（GRBM_*）永不可作触顶证据。非空洞则**立即
停**（走 §3 收尾，不等 3 轮）；空洞或 TASK.md 里压根没声明该谓词，则该报告
不算数，此轮按普通 progress/stale 归类，并在下一轮 prompt 里点名要求要么补
对照实验、要么撤回触顶结论。

## 1. Bootstrap（主会话亲自做，或开一个 agent 做——只用一次）

### 1.-1 入口：slash 命令参数就是三要素

理想用法是用户一次给全：

```
/kernel-opt <kernel 的绝对路径或名字> [生产形状等补充说明]
```

参数解析成三要素：

- **origin**：参数里的路径/名字。是路径就直接用；只是名字（如
  `sol_attn_exact`、`gemm_wmma`）就先在工作区/上游 repo 里定位到具体文件，把
  解析出的绝对路径**复述给用户确认**再动手——名字可能有歧义（同名 kernel 在
  多个 repo / 多个 stage 文件里出现）。
- **brief**：参数里的补充说明 + kernel 是什么、为什么值得优化、**生产形状**、
  验收关注点。生产形状最不能缺——用户没给时，从 origin 的调用点/模型结构推导
  出候选值，**列给用户确认**，不要自己猜定。
- **sandbox**：按 origin 的 kernel 名推导成 `<名字>-opt`（放在用户指定的工作区
  根下，尚不存在）；推导出的路径同样复述确认。

**参数缺生产形状时，推导 + 确认这一步不能省**——bench 测错形状，后面所有数字
都失真。其余两要素（origin 路径、sandbox 名）在参数明确时可直接用，不必再问。

### 1.0 机器档案（先做，注入后续一切）

探测并写成一小段文字，之后 bootstrap 的 build 命令、TASK.md 命令表、以及每个
轮换 agent 的 prompt 尾部都逐字带它——机器无关性就靠这一层。**不要在档案里写
任何你从别台机器记来的具体值——每一项都在目标机上实测**。按下表逐项探，把
「怎么探」跑出真实结果填进档案：

| 项 | 怎么探（在目标机上跑这些，把结果填进档案） |
|---|---|
| arch + wave size | `rocminfo \| grep -i gfx` 取 gfx 名；查该 arch 属 RDNA（wave32）还是 CDNA（wave64）——这决定 hipcc 宏与 kernel 内 `__shfl`/协作组假设 |
| GPU 数/显存/当前占用 | `rocm-smi`（数量、每张 VRAM 总量/已用、当前 util%）→ 据此选一张**空载卡**钉为校准卡，记下卡号 |
| GPU 访问纪律 | 读项目/全局 CLAUDE.md；`groups` 看是否在 render 组；判断 GPU 命令是否需包裹（权限组 `sg`、容器、或直接跑）。**不假设任何机器的包裹方式——没有就直跑** |
| 可见性变量 | 该机器用 `ROCR_VISIBLE_DEVICES` 还是 `HIP_VISIBLE_DEVICES` 钉卡（ROCm 版本相关） |
| 工具链 | `which hipcc rocprofv3 rocminfo rocm-smi`、`hipcc --version`、`cat /opt/rocm*/.info/version` 确认 hipcc/rocprofv3 可用及 ROCm 版本；缺失则停手报告 |
| python/venv（仅 Triton flavor 需要） | 用 origin 所属项目的 venv python 全路径；**PATH 必须带该 venv/bin，否则 JIT/后端静默回退** |
| 知识库 skill | 列出目标机已安装的 kernel 知识库 skill（如有）；有则记 skill 名，无则在档案里写「无」（候选 agent 在 ITERS wiki 栏如实留空，不编造） |
| profiler 计数器 | **先特征化再信任**（见下方警示）：空 kernel / memset / 单块延迟链对照，登记哪些计数器可信、哪些空洞 |

两条跨 arch 的通用警示（撞到时补记进 TASK.md 的 Profiling/命令表节）：

- **busy 率类计数器（如 GRBM_*）可能是空洞证据**：它们在某些 arch 上对空 kernel /
  memset 也读满（是队列级 busy 而非 CU 饱和），不能当 CU 占用率或触顶证据。**任何
  计数器在用作触顶/瓶颈证据前，先在目标机上用空 kernel + memset 对照特征化**；
  哪些空洞随 arch 变，以目标机实测为准。
- **wave size**：RDNA `WARP_SIZE`=32、CDNA 64，影响 hipcc 宏与 kernel 内
  `__shfl`/协作组假设——跨 arch 移植 kernel 时这是头号编译/正确性问题来源。
  以 `rocminfo` 实测的 wave size 为准，不凭印象。

### 1.1 建沙箱

目录结构（HIP 与 Triton 两种 flavor；`(冻结)` 标注冻结清单成员，其余为候选可改的
载体；`<...>` 按任务命名）：

**HIP flavor**

```
<sandbox>/
  build.sh              # hipcc --offload-arch=<arch> -std=c++17 -O3 -I include -I src
  test.hip  -> ./test   # 正确性（冻结）：数值层 + 形状覆盖 + golden hash
  bench.hip -> ./bench  # 性能（冻结）；iter-0 编译后拷一份 bench_base（.gitignore）
  race.hip  -> ./race   # 独立大 grid 确定性探针（冻结）
  TASK.md  ITERS.md  .gitignore
  src/
    <kernel>.hip        # ← 载体：候选只改这里
    harness.hpp         # (冻结) host 驱动
    ref.hpp             # (冻结) 独立参考实现（CPU fp32/fp64）
  include/
    architecture_config.h  # (冻结) arch/dtype 配置头
    <其它被 include 的只读头>  # (冻结)
```

**Triton flavor**

```
<sandbox>/
  kernel_baseline.py    # 上游逐字拷贝，仅 import 改写（冻结）
  kernel_v1.py …        # ← 载体：每 candidate 一个新文件，旧文件不动
  test.py  bench.py     # importlib 按模块名加载（冻结）
  TASK.md  ITERS.md  .gitignore
```

Triton 无 build 步骤（JIT）；测试/基准固定用上游 venv 的 python 全路径调用。

读 `templates/TASK.md` 与 `templates/ITERS.md`，按其骨架创建：

1. **逐字拷贝基线**：kernel 源从 origin 原样拷入，机械适配仅限 include 扁平化、
   import 改写，逐条记录（commit message + TASK.md 溯源节）。
2. **flavor 判定**：HIP（build.sh + hipcc `--offload-arch=<档案 arch>`）或
   Triton（kernel_baseline.py 冻结 + 每 candidate 一个 kernel_vN.py 新文件；
   Triton 的 arch 由 JIT 从当前设备推导，不显式传）。
3. **写冻结 harness**：test（数值层 + 形状覆盖 + golden hash + 大 grid 确定性
   探针——探针≥8 reps、毒化输出、位级比对，形状必须含 ≥8×8 块的大 grid，
   因为 sync race 只在大 grid 显现）、bench（生产形状、median-of-medians、
   交错 A/B 支持、json 落盘）、race 独立探针（HIP: race.hip）、TASK.md、
   ITERS.md。**冻结清单** = 载体之外的一切，写进 TASK.md 并给出逐字冻结校验
   命令（`git diff --exit-code -- <清单>`）。
4. **基线全绿**：build、test、race 探针在空载卡上跑通；跑 sandbox-vs-origin
   文件级 diff，确认只有已记录的机械适配。
5. **HIP flavor 立刻把 iter-0 编译的 ./bench 拷成 bench_base**（交错 A/B 的
   基线载体，写入 .gitignore；后续 candidate 重建不得动它——候选改了 kernel 后
   ./bench 会变，基线必须活着）。
6. `git init` + `iter-0: baseline` commit（溯源 + 适配清单，结尾
   `Co-Authored-By: Claude Code <noreply@anthropic.com>`）。
7. **校准**：用候选将用的同款交错纪律把基线 bench 跑 3 轮（HIP: ./bench_base
   三次；Triton: `bench.py kernel_baseline` 三次——Triton 注意 PATH 必须带上游
   venv/bin，否则 JIT/后端会静默回退到非目标实现），主指标取中位数，3 轮
   spread → noisePct 写进 TASK.md 终止章节；完整基线表写 ITERS.md iter-0；单独
   `calibration` commit。**校准前必须先特征化 profiler 计数器**（见 §1.0 警示）。
8. **突变测试**（harness 强度验证）：在 `/tmp` scratch 拷贝上对载体施加 4 类
   突变（sync/boundary/coverage/numeric，恰好一处），重建跑全套测试——
   **每类都必须被抓住**。有存活 = harness 盲区，只强化 harness（不碰载体），
   独立「harness 补强」commit，重试存活类别。scratch 命令写成
   `rm -rf <dst> && cp -r`，别依赖记忆清理。sync 类独占静置卡跑满 reps。
9. 把 noisePct、冻结清单、验收规则复述给用户，让用户 review 沙箱后再开始轮换。
   **bootstrap 只建一次**；已有沙箱继续迭代直接从 §2 进入。

## 2. 轮换制（campaign 主循环）

### 启动一轮

用 Agent 工具开 **subagent_type: "general-purpose"** 的后台 agent；**模型在
`"opus"` 与 `"sonnet"` 之间轮换**（传 `model:` 参数）——换模型正是为了换视角，
连续同模型容易在同样的假设上打转；这两个模型档位足够拉开视角差异，引入更低
档位收益递减。一轮一个 agent，不并行（A/B 钉同一张卡，且冻结校验要求工作树
串行）。

prompt 按下面的模板写（**不要**把 TASK.md 内容贴进去——agent 自己会读）：

```
你是沙箱 <绝对路径> 的本轮优化者（轮 R<k>）。

先读：
1. 沙箱 TASK.md（先读首节「目录里的进度不是你的」，按其复现最近 accepted 状态：
   git status 干净 → 冻结校验 → test → 一次 bench 对 ITERS.md 数字）
2. ITERS.md 全部条目（哪些方向被否决过、为什么——别重复已否决方向）

然后按 TASK.md 的工作流做 candidate（1-3 个，做不完把剩余假设写进汇报）：
立假设 → 查知识库（机器档案给了知识库 skill 名时，用 Skill 工具调它，如
`KernelWiki`；查不到条目如实记「无」，不编造）→ 只改载体 →
冻结校验 → test → race 探针 → 同 session 交错 A/B bench（before = 同 session
重测的基线）→ 取证 → ITERS.md 追加 wip 条目 → commit 或 revert（验收规则见
TASK.md 性能判定节，noisePct 是硬门槛）。

硬规则：
- test 挂 = candidate 作废，bench 不跑，revert 后立下一个假设或结束本轮。
- 一个 candidate 一 commit（accepted）或一 revert（rejected），工作树始终保持
  干净或已提交。
- **命令一律用 TASK.md 命令表的逐字版本，不自造、不"顺手修"。命令本身坏了
  （路径错/venv 换了/工具缺失）= 停手按 envBlocked 汇报——harness 只能由
  主会话在 candidate 流程之外修（独立 harness commit），候选 agent 永不碰。**
- 复现失败（基线数字对不上）：不优化，先查原因，如实汇报 drift。
- 环境被占（另一张卡也忙 / 依赖坏）：停手汇报 envBlocked，不硬跑。
- profiling 只能用 TASK.md Profiling 节里已特征化的计数器（空洞清单上的禁用），
  包裹命令照抄命令表。

结束本轮时，最终汇报必须包含：
(a) 本轮每个 candidate 的结局（accepted/rejected + 一句话理由）；
(b) 裁决二选一：「有进展/还有具体假设没试」（列出假设清单）或
   「没有进一步空间」（必须附证据：结构空间扫描结果 / 触顶对照实验 / 已否决
   方向清单——光说"想不到"不算）；
(c) 当前 git log 最近 3 条与 git status。
GPU 纪律与机器档案：<逐字粘贴 §1.0 的档案，含访问纪律包裹方式与校准卡号>。
```

**一轮 1-3 个 candidate** 交给 agent 自裁量——单个 agent 上下文有限，塞太多会
让取证质量下降；但一轮只做 1 个又让换模型的视角多样性来不及发挥。

### 一轮结束后主会话做四件事

1. **验收**：`git -C <sandbox> status` 干净、log 与汇报一致、ITERS.md 新增条目
   数字与 bench json 对得上（抽一个数复核即可；对不上就查——agent 夸大数字是
   已知风险）。
2. **裁决归类**：progress（计数清零）或 stale（计数 +1；「没空间」判断无证据
   支撑则打回不算，下一轮 prompt 里点名要求补结构扫描）。
3. **决定**：满足任一终止条件 → §3 收尾——(a) stale 计数到 3，或 (b) 本轮
   报告**非空洞触顶**（按 §0 查过对照实验登记）。否则**换成另一个模型**
   （opus↔sonnet）启动下一轮，prompt 里附上上轮的简短结论和「上轮说没空间
   被打回」之类的上下文。
4. 转述给用户：本轮结局 + 当前累计收益 + stale 计数。

### 中途死亡 / 环境故障

agent 死了、汇报 envBlocked、或**长时间无进展**（模型 API 连接失败：transcript
末尾是 `API Error: Connection refused`、文件几分钟不更新、工作树干净——这是
模型侧故障不是 GPU 故障）时：

1. 先 `rocm-smi` 看占用、看沙箱 `git status`。
2. 树脏了就机械恢复（`git checkout -- . && git clean -fd`，Triton 另删多余
   kernel_vN.py）；树干净则无需恢复。
3. 重启一轮。**模型 API 死的就换另一个模型重启**（opus 死了换 sonnet，反正
   轮换制本就要求换模型），别傻等同模型恢复。
4. **这类轮一律不计入 stale 计数。**

## 3. 收尾（连续 3 stale 后）

主会话亲自做（读 `templates/ITERS.md` 末尾的终止小结模板）：

1. ITERS.md 末尾追加**终止小结**（追加式，不改旧条目）：命中哪条终止条件
   （连续 3 轮无空间 / 触顶谓词命中——后者要写明谓词内容与其非空洞对照实验）、
   终态数字 vs iter-0 基线表、总收益、证据指针、剩余空间一句话。
2. `termination` commit。
3. 转述给用户：accepted 迭代列表、总收益、剩余空间。优化合回原仓库由用户
   另行发起，不属于本 skill。
