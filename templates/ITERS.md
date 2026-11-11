# ITERS — <kernel 名> candidate log

每个 candidate 一条，追加在末尾，不改旧条目；更正用「更正 — iter-N …」追加段。
收益用 bench 数字，瓶颈用 profiler / 反汇编证据；无证据的推测只写「假设」栏。
规则见 TASK.md。冻结校验每次改动后必跑。

## 条目模板

```
## iter-N — <一句话方向>
- 状态: accepted | rejected | wip
- 层: <t1 DSL / t2 native / t3 low-level>
- commit: <hash；rejected 写「已 revert」>
- 假设: 改什么、为什么预期更快、预期哪段受益
- wiki: KernelWiki 条目（查不到写「无」）
- 改动: 具体文件和位置
- 冻结校验: FROZEN OK
- 正确性: test 全 PASS?（max_rel 或 bitwise）；race 全 OK?
- 性能 (before→after, before 为同 session 重测基线):
    <各指标数字 + relPct>
- 证据: profiler / 反汇编 / 探针的前后值
- 结论: 采纳/回退理由（对照 TASK.md 验收规则）
```

## 层级转移条目模板

```
## tier-N — <层名> baseline（从 tier-M accepted 翻译）
- 状态: accepted (tier baseline)
- 上游层: tier-M iter-K commit <hash>——翻译蓝本
- 下穿归因: <profiler / 反汇编证据：剩余差距来自上层哪个机制（spill、占用率、
  异步拷贝/流水、特定指令、调度等）；为何不是物理/算法上限。无证据不换层>
- 翻译说明: 保留的算法/分块/流水/融合/布局决策；低层重新表达的部分
- 跨层 A/B（同 session）: 上层 accepted <值> → 新层基线 <值>（relPct；允许
  初版落后，注明差距）
- 正确性: test 全 PASS?（golden 重录说明）；race 全 OK?
- 新层校准: 同 session 3 轮 spread = <noisePct>%
- 结论: 新层可继续优化；若到顶仍落后则整体 best 仍是上层
```

---

## iter-0 — baseline（t<起始层>，沙箱建立 + 校准）

- 状态: accepted (baseline)
- 层: <t1 DSL / t2 native / t3 low-level>
- commit: <baseline commit hash>
- 溯源: <上游 repo + tag/commit；机械适配逐条列出>
- 校准: 同 session 基线 bench 3 轮，主指标中位数 <值>，spread = <noisePct>%
  → 写入 TASK.md；次要指标 spread 同样记录。
- profiler 计数器特征化: <可用与空洞清单；换平台 / arch / 层必须重做>
- 基线完整表:

  | 指标 | 值 | 备注 |
  |---|---|---|
  | <主指标> | <值> | <方向/聚合> |
  | <次要指标…> | <值> | <回归上限> |

- bench 原始落盘: <json 路径或终端输出位置>

---

<!-- 之后的 iter-N 条目由 candidate 追加在此线之下 -->

<!--
## 终止小结（主会话收尾时追加）

- 命中终止条件: <最低适用层连续 N 轮无空间 / 非空洞触顶谓词命中>
- 各层旅程: <t1/t2/t3 是否进入、各层最终主指标与收益、跳过层的 N/A 原因>
- 最终采纳层: <整体 best 所在的层与实现/commit>
- 终态 vs iter-0: <主指标 before→after 表格 + 总收益 %>
- accepted 迭代: <iter 号 + 方向 + hash，逐条>
- 证据指针: <关键 profiler 对照 / 反汇编落盘位置>
- 剩余空间: <结构空间是否扫尽；触顶时写已达哪种物理上限>
-->
