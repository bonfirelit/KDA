# ITERS — <kernel 名> candidate log

每个 candidate 一条，**追加在末尾，不改旧条目**。事后更正以「更正 — iter-N …」
追加段处理，不回写原条目。收益用 bench 数字，瓶颈用 profiler / 反汇编证据；
**无证据的推测只能写在「假设」栏，验证后回填**。规则见 TASK.md。

冻结文件校验每次改动后必跑（逐字命令见 TASK.md 可改/冻结节）。

## 条目模板

```
## iter-N — <一句话方向>
- 状态: accepted | rejected | wip
- 层: <t1 DSL / t2 native / t3 low-level>
- commit: <hash；rejected 则写「已 revert」>
- 假设: 改什么、为什么预期更快、预期哪层/哪段受益
- wiki: KernelWiki 支撑条目（查不到写「无」，不编造）
- 改动: 具体改了哪些文件哪里
- 冻结校验: FROZEN OK（贴 git diff --exit-code 输出）
- 正确性: test 全 PASS?（max_rel 或 bitwise）；race 探针全 OK?
- 性能 (before→after, 同 session 重测基线为 before):
    <各指标数字 + relPct；注明 before 是同 session 重测值而非 iter-0 历史值>
- 证据: profiler / 反汇编 / 探针的具体计数器前后值
- 结论: 采纳/回退的理由（对照 TASK.md 验收规则逐条）
```

## 层级转移条目模板

```
## tier-N — <层名> baseline（从 tier-M accepted 翻译）
- 状态: accepted (tier baseline)
- 上游层: tier-M iter-K commit <hash>——翻译蓝本
- 下穿归因: <profiler / 反汇编证据：剩余差距来自上层表达力/codegen 的哪个具体
  机制（spill、占用率、异步拷贝/流水、特定指令、指令调度等）；为何不是物理/
  算法上限。无归因证据不允许换层>
- 翻译说明: 保留的算法/分块/流水/融合/布局决策；低层重新表达的部分；
  只属于上层的约束（如 DSL tile 形状）
- 跨层 A/B（同 session）: 上层 accepted <值> → 新层基线 <值>（relPct；允许初版
  落后，注明差距与原因）
- 正确性: test 全 PASS?（golden 重录说明）；race 全 OK?
- 新层校准: 同 session 3 轮 spread = <noisePct>%
- 结论: 新层可继续优化 / 与上层差距说明 / 若到顶仍落后则整体 best 仍是上层
```

---

## iter-0 — baseline（t<起始层>，沙箱建立 + 校准）

- 状态: accepted (baseline)
- 层: <t1 DSL / t2 native / t3 low-level>
- commit: <baseline commit hash>
- 溯源: <上游 repo 路径 + tag/commit；机械适配逐条列出（include 扁平化、import
  改写等），每条一行>
- 校准: 同 session 基线 bench 3 轮，主指标中位数 <值>，3 轮 spread = <noisePct>%
  → 写入 TASK.md 终止章节作为噪声阈值；次要指标 spread 同样记录。
- profiler 计数器特征化: <可用清单与空洞清单——空 kernel / memset 对照结果；
  换平台 / arch / 层必须重做>
- 基线完整表:

  | 指标 | 值 | 备注 |
  |---|---|---|
  | <主指标> | <值> | <方向/聚合> |
  | <次要指标…> | <值> | <回归上限> |

- bench 原始落盘: <json 路径或终端输出位置>

---

<!-- 之后的 iter-N 条目由 candidate 追加在此线之下 -->

<!--
## 终止小结（收尾时由主会话追加，模板）

- 命中终止条件: <二选一——(a) 最低适用层连续 N 轮无空间（列出这 N 轮的结论
  一句话）；(b) 触顶谓词命中（写明谓词内容 + 其非空洞对照实验：空 kernel/memset
  不触发它）>
- 各层旅程: <t1/t2/t3 是否进入、各层最终主指标与主要收益、跳过的层写 N/A 原因>
- 最终采纳层: <整体 best 所在的层与实现/commit>
- 终态 vs iter-0: <主指标 before→after 表格 + 总收益 %>
- accepted 迭代: <iter 号 + 一句话方向 + hash，逐条>
- 证据指针: <关键 profiler 对照 / 反汇编落盘位置>
- 剩余空间: <一句话：结构空间是否扫尽、还有哪些被否决但可能换个前提成立的方向；
  触顶停止时写「已达 <哪种> 物理上限，剩余空间不在本 kernel」>
-->
