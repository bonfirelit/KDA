# TASK — <kernel 名> 优化（<arch> / ROCm <版本>）

> 模板用法：把 `<...>` 全部替换为实际值；方括号 `[...]` 是可选段，不需要就删。
> 章节顺序固定（沿用范例先例），不要增删一级章节——轮换 agent 依赖这个结构导航。

## 先读这个：目录里的进度不是你的

本目录由多个 agent 多轮迭代；`ITERS.md` 的一切（含 iter-0 基线）都可能错。
接手第一件事不是优化，是**复现最新已采纳状态**：

```bash
git status                          # 须干净（冻结文件无改动）
<build 命令>                         # HIP flavor 才需要
<test 命令>                          # 全 PASS（含 golden hash）
<bench 命令>                         # 对照 ITERS.md 最后 accepted 条目数字
```

对不上就先查原因（环境 / 冻结文件被改 / 该结论本身错了），把差异写进 ITERS.md
修正或作废对应条目；**复现不了的结论当不存在**。

## 目标（含基线数字）

- **kernel 是什么**：<一句话：做什么计算、输入输出布局>。
- **为什么值得优化**：<占 GPU 时间的比例 / 端到端里的位置>。
- **生产形状（bench 就测这个）**：<B/T/H/D、layout、dtype、稀疏度等——最不能缺>。
- **基线数字（iter-0）**：<主指标值 + 次要指标值，来源：校准>。
- **验收关注点**：<除了主指标还在乎什么——功耗？别的形状不能回归？>

## kernel 背景与生产调用约定

<上游怎么调用它：入口函数、dispatch 条件、transposed-view 之类的真实内存布局、
哪些 stage 生产实际跑哪些不跑。这段决定 bench 的形状与布局是否贴生产——
写错了后面所有数字都失真。>

## 可改 / 冻结

**载体（候选只能改这里）**：
```
<kernel 实现文件/头，或 Triton 的 kernel_v*.py 模式>
```

**冻结清单（测什么的一切）**：
```
<test / bench / race 探针 / host harness / 参考实现 / arch·dtype 配置头 /
 生产不跑但参与编译的 stage 文件 / 入口·plan 文件>
```

冻结校验命令（每个 candidate 改动后必跑，输出贴 `FROZEN OK`）：
```bash
git diff --exit-code -- <冻结清单，逐字列出>
```

## 正确性门禁

三层，单层会被骗（见 ITERS.md 引用的 race 事故教训）：

1. **数值层**：参考实现独立于被测 kernel（CPU fp32/fp64 `ref()`，或生产构建的
   二进制 dump）。容差按路径写死。<如 rel ≤ 2e-3；scalar-scale 路径要求 bitwise>。
   **判据选型警告（bf16/相消输出的坑）**：若 kernel 输出是两项相消型
   （如 rope 的 cos·a−sin·b、dot 的部分和），ref 值可精确为 0——此时**相对误差
   （除零放大）与 ULP（bf16 在 0 附近隔所有 subnormal，1 尾数差=数万有序整数差）
   双双假阳性**。这种情况必须用 **abs+rel 组合**（`|a−b| ≤ atol + rtol·|b|`，
   atol 兜零点附近、rtol 管大值），不要用单一 rel 或单一 ULP。fp32 输出无此坑，
   可用 rel 或 ULP。
2. **形状覆盖**：生产形状 + 尾部/边界（K 非 tile 整倍数、不规则 M/N、掩码/转置
   布局按 kernel 语义）+ 小形状对照 + 极端值。**必须含至少一个 grid ≥ 生产块数
   （≥8×8 块）的形状**——race 只在大 grid 显现。
3. **位级 golden hash 层**（kernel 确定时）：每个 case 输出字节 FNV-1a 与硬编码
   常量比对；新 case 先跑基线记录 hash（`0 = RECORD ME`）。两个实操要点：
   (a) **hash 只钉形状固定的 case**——形状随 CLI/参数变的 case（如生产形状）golden
       留 0 跳过强比对，由 abs+rel 层把守，否则换参数就误挂。
   (b) **golden 会挡合法改变累加顺序的优化**（数值仍对但 hash 变）——候选遇到时
       按 RECORD ME 协议重录，且**数值仍须先过 abs+rel 层**才允许重录，在 ITERS
       条目注明重录理由。
4. **大 grid 确定性探针**（独立可执行）：小 grid 对照 + 8×8/16×16 interior +
   一个生产真实形状；每形状 R≥8 reps、每次 launch 前毒化输出 buffer、全部 rep
   位级比对、rep-0 另对参考。判定：reps_differ>0 → RACE；ref 对不上 → WRONG。
   **跑在静置卡上，不与其它 GPU 负载并发。**

## 性能判定

- **主指标**：<名字、方向（higher/lower better）、聚合（median-of-medians）>。
- **次要指标回归上限**：<逐个列出名字 + maxRegressionPct + 方向——三者缺一不可，
  方向不写清「回归」二字没有定义>。
- **噪声阈值（校准实测）**：同 session A/A spread = <noisePct>% —— 主指标提升
  **严格大于**此值才可能采纳。**跨 session 噪声远大于同 session（本类机器实测
  量级差 10 倍以上），跨 session 数字不可比**。
- **交错 A/B 机制**：<HIP：iter-0 编译的 ./bench 拷成 ./bench_base 与候选 ./bench
  同 session 交替；Triton：同进程 `bench.py kernel_baseline kernel_vN` 多模块。
  candidate 的 before 一律取同 session 重测值，不取 iter-0 历史值>。

## 工作流（每个 candidate 严格按序，任一步失败即停）

1. 复现基线（见首节）→ 确认树干净。
2. **立假设**：改什么、为什么预期更快、预期哪层/哪段受益。
3. **查知识库 skill**：机器档案里给了知识库 skill 名（如 KernelWiki）时，用
   Skill 工具按名调用它查询；引用条目记入 ITERS 的 wiki 栏；查不到或本机无此
   skill 时如实留空，不编造。
4. **只改载体**。HIP：编辑 kernel 文件后 `<build 命令>`；Triton：新建
   `kernel_v<N>.py`（旧文件不动）。
5. 冻结校验（命令见上节）。
6. `<test 命令>` 全 PASS——**挂了 = candidate 作废，bench 不跑**。
7. race / 确定性探针全 OK。
8. **同 session 交错 A/B bench**：先重测基线，再测候选，交替。
9. 取证（profiler 计数器 / 反汇编 / 带宽核算）。
10. ITERS.md 追加 iter-N 条目（状态 wip）。
11. commit（accepted）或 revert（rejected）——一个 candidate 一个动作。

## 终止条件

满足任一即停：

1. **连续 3 个 candidate**（或连续 3 个 agent 轮，取外层编排用的口径）主指标
   提升都在噪声阈值内（阈值来自校准，不是硬编码）。
2. **触顶谓词命中**——且该谓词已用对照实验证明非空洞：<写出本 kernel 的谓词，
   例「decode weight GB/s ≥ 0.9×实测 HBM 有效带宽」。注意：busy 率类计数器
   （如 GRBM_*）对空 kernel 也读 100%，永不可作触顶证据>。
3. **红线**：<按任务定，如「次要指标回归超限连续 N 次」>。

## 命令表

> 逐字可粘。GPU 访问纪律按机器档案填：包裹前缀（若需，如权限组/容器包裹）与
> 钉卡变量（`ROCR_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES`，按机器档案实测）都
> 取自机器档案，**不假设任何机器的包裹方式**。写死后不再改。

| 用途 | 命令 |
|---|---|
| 选卡 | `rocm-smi`（选空载卡，钉为校准卡 <gpu-id>） |
| build | `<HIP: hipcc --offload-arch=<arch> ...；Triton: 无>` |
| test | `<逐字>` |
| race 探针 | `<逐字>` |
| bench | `<逐字，含 --export json>` |
| 冻结校验 | `git diff --exit-code -- <清单>` |
| 知识库 | `<知识库 skill 名（如 KernelWiki），用 Skill 工具调用；无此 skill 则写「无」>` |
| profiler | `<rocprofv3 包裹命令>`（注意：rocprofv3 `--pmc`/`--sys-trace` 输出是 SQLite `.db`，
  不是 CSV——查询用 python sqlite3，见 Profiling 节） |

## Profiling

- **本机已特征化、可用的计数器**：<逐个列出 + 特征化方法>。
- **已知空洞、不可作证据的计数器**：<逐个列出 + 空洞的对照实验现象，
  例「GRBM_GUI_ACTIVE 对空 kernel / memset 也读 100%，是队列级 busy 而非 CU
  饱和」>。
- 换 arch 后**必须重新特征化**（空 kernel / 单块延迟链 / memset 对照），空洞
  清单随 arch 变。
- **rocprofv3 输出格式**：`--pmc <counters>` 与 `--sys-trace` 的产物是 SQLite `.db`
  （含 `rocpd_info_pmc_*` 计数器元数据表 + `rocpd_pmc_event_*` 值表），不是 CSV。
  查询（任一 python3 + sqlite3 标准库；表名后缀随 run 变，先
  `SELECT name FROM sqlite_master WHERE type='table'` 探）：
  ```sql
  SELECT i.name, SUM(e.value), COUNT(e.value)
  FROM rocpd_pmc_event_XXX e JOIN rocpd_info_pmc_XXX i ON e.pmc_id=i.id GROUP BY i.name;
  ```
