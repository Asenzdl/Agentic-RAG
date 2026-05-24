---
title: ADR 书写规则 (ADR Writing Rules)
description: AI 在 DR 关联时创建 ADR 骨架，后续渐进填充。ADR 是决策的稳定锚点，跟踪生命周期。
---

## AI 行为约束

### 创建时机
- 第一条 DR 关联到某个 ADR 编号时，立即创建 ADR 骨架
- 骨架包含 ADR 编号、Status、来源 DR 链接，正文暂为占位
- 后续 DR 关联时，AI 读取 ADR 当前内容并**增量补充**（不改已有文字，只追加）

### 来源与引用
- 头部「来源 DR」列出所有贡献给本 ADR 的 DR，用逗号分隔
- 这是 DR → ADR 双向链接的 ADR 端
- ADR 的内容必须可追溯到来源 DR（必要时标注对应 DR 编号）

### Status 生命周期
- `Active` — 当前有效
- `Superseded by ADR-NNN` — 已失效，被 NNN 号 ADR 替代
- 过渡：由 AI 在写入新 DR 时检测到条件变化→更新 ADR Status

### 信息保真度
- 所有写入 ADR 的信息必须可回溯到来源 DR，不得新增对话中未出现的内容
- ADR 的「Context」和「Decision」可以从 DR 中提取浓缩，但前提是不丢失决策的关键约束
- C0 约束在 ADR 层需解释为「可聚焦，不可扭曲」——聚焦于该 ADR 管辖的决策点，DR 中的其他噪声可忽略，但本决策点的关键细节不可丢失

### 文件位置

ADR 存放于 `docs/adr/{module}/` 下，与 DR 混排。详见 `DR_TEMPLATE.md` 的目录结构示例。

### 命名规则
`ADR-NNN-简短描述.md`

- `NNN` = 全局顺序编号，从 001 开始，永不重复
- 如 `ADR-001-retriever-architecture.md`

---

## ADR 模板

```markdown
# ADR-NNN: {决策标题}

- **Status**: Active <!-- 或 Superseded by ADR-NNN -->
- **Date**: YYYY-MM-DD
- **来源 DR**: DR-YYYYMMDD-NN [, DR-YYYYMMDD-NN]

## Context

{触发场景。关联此 ADR 的所有决策记录中，与本决策直接相关的上下文。}

## Decision

{最终选择的方案及核心理由。可聚焦提取，不得扭曲。}

## Consequences

{正向 / 负向 / 中性影响}

**前提假设：**
{决策成立依赖的条件}

**重审信号：**
{条件变化后需要重审的信号}
```
