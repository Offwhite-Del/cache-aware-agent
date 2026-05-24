# CacheAwareAgent — 零依赖 Python 缓存感知 Agent

> 利用 DeepSeek 前缀缓存实现 80-90% 的 API 成本节省，纯 Python 标准库实现。

## 核心原理

DeepSeek 服务端会缓存已计算的前缀 KV cache。
如果两次请求的 prompt 从第一个 token 开始**字节级完全一致**，
缓存命中 token 的价格仅为普通 token 的 **10%**。

CacheAwareAgent 通过三条铁律最大化缓存命中率：

```
铁律1：三段式上下文     → system + 仅追加日志 + 运行时暂存（不进API）
铁律2：字节稳定前缀     → system prompt + tools 会话期间字节级不变
铁律3：仅追加原则       → 历史消息只能append，不能修改/删除/重排
```

## 性能数据

| 指标 | 值 |
| --- | --- |
| 缓存命中率（预热后） | **99.82%** |
| API成本节省 | **89.8%** |
| 日调用量 | 435M token |
| 外部依赖 | **零**（仅 Python 标准库） |

## 用法

```python
from cache_aware_agent import CacheAwareAgent

agent = CacheAwareAgent(
    system_prompt="你是一个AI助手",
    tools=[...],
    api_key="sk-xxx",
)

agent.add_user_message("今天修了什么bug？")
reply = agent.send()
print(reply)
print(agent.report())  # 缓存命中报告
```

### 为什么不是 TypeScript？

| | CacheAwareAgent | Reasonix (TS) |
| --- | --- | --- |
| 依赖 | 零（标准库） | npm + TypeScript |
| 运行环境 | 任何有 Python 的地方 | 需要 Node.js |
| 缓存策略 | 前缀缓存 + 三段式 | 多层缓存（LRU/前缀） |
| 适用场景 | 嵌入式/工控/边缘设备 | 高性能 Serverless |

**理念不同**：Reasonix 是高性能框架，CacheAwareAgent 是极简工具——在工控机、树莓派、嵌入式设备上也能跑。

## 三条铁律详解

### 铁律1：三段式上下文

```
┌─────────────────────────┐
│  不可变前缀             │  ← system prompt + tools（永不修改）
│  (system prompt + tools)│
├─────────────────────────┤
│  仅追加日志             │  ← user message + assistant reply + tool result
│  (历史消息)             │     只能 append，不能改/删/重排
├─────────────────────────┤
│  运行时暂存             │  ← 思考过程、中间结果（**不进API**）
│  (scratchpad)           │     只在内存中，不参与请求构造
└─────────────────────────┘
```

### 铁律2：字节稳定前缀

System prompt + tools 的计算顺序是确定性的：
- System prompt 在初始化后不再修改（除非归档）
- Tools 按 function name 排序（确定性顺序）
- 整个会话生命周期字节级一致

### 铁律3：仅追加原则

历史消息只能追加，不能修改、删除或重排。
唯一例外：归档时更新 system prompt 并清空日志（设计允许的唯一修改点）。

## 归档机制

当消息数超过 300 条或预估 token 超过 80K 时自动归档：
1. 用 DeepSeek API 生成会话摘要
2. 将摘要追加到 system prompt
3. 清空历史消息

归档后缓存前缀改变（system prompt 变化），预热约 1-2 轮后恢复高命中率。

## 安装

```bash
# 不需要安装，直接复制文件即可
git clone https://github.com/Offwhite-Del/cache-aware-agent
```

## 依赖

**零外部依赖**。仅使用 Python 标准库：
- `json`, `gzip`, `time`, `logging`, `copy`
- `urllib.request`（HTTP 客户端）

## 许可证

MIT
