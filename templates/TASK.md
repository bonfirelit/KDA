# TASK — <kernel 名> 优化（<平台> <arch> / <工具链> <版本>）

> 模板用法：替换 `<...>`；章节顺序固定，不增删一级章节。

## 先读这个：目录里的进度不是你的

多 agent 多轮迭代，ITERS.md 的一切都可能错。先复现当前层最新 accepted 状态：

```bash
git status
<build 命令>      # 原生编译才需要
<test 命令>       # 全 PASS（含 golden hash）
<bench 命令>      # 对照 ITERS.md 当前层最后 accepted 条目
```

对不上先查原因（环境 / 冻结文件被改 / 结论本身错），差异写进 ITERS 修正或
作废；复现不了的结论当不存在。

## 目标（含基线数字）

- **kernel 是什么**：<一句话：计算、输入输出布局>。
- **为什么优化**：<占加速器时间的比例 / 端到端位置>。
- **生产形状（bench 就测这个）**：<B/T/H/D、layout、dtype、稀疏度等>。
- **基线数字（iter-0）**：<主指标 + 次要指标，来源：校准>。
- **验收关注点**：<除主指标外还在乎什么——功耗？其它形状不能回归？>。

## 层级计划（tier ladder）

- **起始层**：<t1 DSL / t2 native / t3 low-level>——依据 origin 的实现语言；
  origin 是 CUDA/HIP 就从 t2 开始，不绕 DSL。
- **规划阶梯**：<如 t1 → t2 → t3；某层不可用时写 `t1 → t2（t3 N/A：证据）`>。
- **最低适用层**：<通常 t3；不可用时写实际最低层 + 机器档案证据>。
- **当前层**：<t1 / t2 / t3>——本文件命令、载体、冻结清单都指当前层。
- **到顶与转移**：当前层连续 3 个 stale 轮或非空洞触顶 → 到顶；主会话先做
  瓶颈归因（剩余差距是否来自当前层表达力 / codegen：spill、占用率、异步拷贝/
  流水、特定指令、调度等）。层级性上限且有更低可用层才转移（SKILL.md §1.2）；
  物理/算法上限或最低适用层到顶则终止。
- **下穿归因（转移时填）**：<profiler / 反汇编证据：剩余差距来自上层哪个具体
  机制；为何不是物理/算法上限。无证据不换层>。
- **各层基线**：

  | 层 | 实现 / commit | 主指标 | noisePct | 备注 |
  |---|---|---|---|---|
  | <t1 DSL> | <iter-0 hash> | <值> | <校准值> | <起点> |
  | <t2 native> | — | — | — | <转移后填> |

## kernel 背景与生产调用约定

<上游调用方式、入口、dispatch 条件、真实内存布局、生产实际跑哪些 stage。>

## 可改 / 冻结

**载体（候选只能改当前层的这里）**：

```
<当前层 kernel 实现文件/头，或 Triton 的 kernel_v*.py 模式>
```

**已到顶层的 accepted 实现（冻结，保留可编译可运行，跨层 A/B 用）**：

```
<上一层路径 + commit> 或「无（当前是起始层）」
```

**冻结清单（测什么的一切）**：

```
<test / bench / race / host harness / 参考实现 / arch·dtype 配置头 /
 生产不跑但参与编译的 stage 文件 / 入口·plan 文件 / 已到顶层的 accepted 实现>
```

冻结校验（每个 candidate 改动后必跑，输出贴 `FROZEN OK`）：

```bash
git diff --exit-code -- <冻结清单，逐字列出>
```

## 正确性门禁

1. **数值层**：ref 独立于被测 kernel（CPU fp32/fp64 或生产二进制 dump），容差
   按路径写死。bf16/相消输出用 `|a−b| ≤ atol + rtol·|b|`，不用单一 rel 或
   ULP；fp32 可用 rel 或 ULP。
2. **形状覆盖**：生产形状 + 尾部/边界（K 非 tile 整倍数、不规则 M/N、掩码/
   转置）+ 小形状 + 极端值；至少一个 ≥8×8 块的大 grid（race 只在大 grid
   显现）。
3. **golden hash**：固定形状 case 的输出字节 FNV-1a 与常量比对，新 case 记
   `0 = RECORD ME`；形状随参数变的 case 留 0。改变累加顺序时按 RECORD ME
   重录，且先过数值层，ITERS 注明理由。
4. **大 grid 确定性探针**：小 grid + 8×8/16×16 interior + 生产形状；每形状
   R≥8 reps、launch 前毒化输出、全部 rep 位级比对、rep-0 对参考。
   `reps_differ>0` → RACE；ref 不符 → WRONG。跑静置卡。

## 性能判定

- **主指标**：<名字、方向（higher/lower better）、聚合（median-of-medians）>。
- **次要指标回归上限**：<名字 + maxRegressionPct + 方向>。
- **噪声阈值（按层独立）**：当前层同 session A/A spread = <noisePct>%，提升
  严格大于才可采纳。不同层不共用；转移后重新校准。跨 session 数字不可比。
- **同层交错 A/B**：<原生：iter-0 的 ./bench 拷成 bench_base 与候选 ./bench
  同 session 交替；Triton：同进程 `bench.py kernel_baseline kernel_vN`。
  before 一律取同 session 重测值>。
- **跨层 A/B**：转移时上层 accepted 与新层基线同 session 交错；新层初版可落后
  但记录差距，目标超过上层；到顶仍落后则整体 best 是上层。

## 工作流（每个 candidate 严格按序，任一步失败即停）

1. 复现基线，确认树干净。
2. **立假设**：改什么、为什么更快、预期哪段受益。
3. **查知识库 skill**：档案有知识库 skill 则用 Skill 工具查询，引用条目记入
   ITERS 的 wiki 栏；查不到或本机无此 skill 如实留空。
4. **只改当前层载体**；不跨层翻译、不改 harness/上一层。原生编译后
   `<build 命令>`；Triton 新建 `kernel_v<N>.py`（旧文件不动）。
5. 冻结校验。
6. `<test 命令>` 全 PASS，否则 candidate 作废、不跑 bench。
7. race / 确定性探针全 OK。
8. **同 session 交错 A/B**：先重测基线，再测候选，交替。
9. 取证（profiler 计数器 / 反汇编 / 带宽核算）；档案有 profiler 报告分析
   skill 则用它解析 report。
10. ITERS 追加 iter-N 条目（wip）。
11. commit（accepted）或 revert（rejected）。

## 终止条件（按层判定）

当前层到顶 → 有更低可用层则层级转移（SKILL.md §1.2），最低适用层到顶才终止：

1. **当前层连续 3 个 candidate**（或连续 3 个 agent 轮）主指标提升都在该层
   噪声阈值内。
2. **非空洞触顶谓词命中**：<本 kernel 的谓词，例「decode weight GB/s ≥
   0.9×实测 HBM 有效带宽」>。busy 率计数器（各平台 SM/队列 busy，AMD 如
   GRBM_*）对空 kernel 也读 100%，永不作证据。
3. **红线**：<如「次要指标回归超限连续 N 次」>——命中即终止整个 campaign。

**整层跳过**：工具链不可用或语义明显不适用时，写 N/A + 证据，视为已到顶。

## 命令表

> 下表是当前层的命令。访问纪律、包裹前缀、钉卡变量（NVIDIA
> `CUDA_VISIBLE_DEVICES`；AMD `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`）
> 按机器档案填。层级转移时由主会话随 `harness: tier-N support` commit 更新；
> candidate 不改。

| 用途 | 命令 |
|---|---|
| 选卡 | `<设备查询命令>`（如 `nvidia-smi` / `rocm-smi`；选空载卡，钉为校准卡 <gpu-id>） |
| build | `<原生：<编译器> <arch flag> ...（如 hipcc --offload-arch=<arch>、nvcc -arch=sm_<xx>）；Triton: 无>` |
| test | `<逐字>` |
| race 探针 | `<逐字>` |
| bench | `<逐字，含 json 落盘参数>` |
| 跨层 A/B | `<上一层 accepted 与新层 impl 同 session 交错命令；仅转移时用>` |
| 冻结校验 | `git diff --exit-code -- <清单>` |
| 辅助 skill | 知识库：<skill 名或「无」>；profiler 报告分析：<skill 名或「无」> |
| profiler | `<包裹命令（如 ncu/nsys、rocprofv3）>`（输出格式按平台：AMD rocprofv3 的 `--pmc`/`--sys-trace` 为 SQLite `.db`，见 Profiling 节） |

## Profiling

- **已特征化的可用计数器**：<逐个列出 + 特征化方法>。
- **空洞、不可作证据的计数器**：<逐个列出 + 对照现象>。
- 换平台 / arch / 层后**必须重新特征化**（空 kernel / 单块延迟链 / memset），
  空洞清单随代码生成路径变。
- **输出格式按平台记**：<如 NVIDIA 用 ncu/nsys 报告；AMD rocprofv3 的
  `--pmc`/`--sys-trace` 产物是 SQLite `.db`（`rocpd_info_pmc_*` 元数据 +
  `rocpd_pmc_event_*` 值），不是 CSV>。
- **ROCm 查询示例**（仅 AMD；表名后缀随 run 变，先查
  `sqlite_master`）：

  ```sql
  SELECT i.name, SUM(e.value), COUNT(e.value)
  FROM rocpd_pmc_event_XXX e JOIN rocpd_info_pmc_XXX i ON e.pmc_id=i.id GROUP BY i.name;
  ```
