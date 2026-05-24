#!/usr/bin/env python3
"""
CacheAwareAgent — 利用 DeepSeek 前缀缓存的极简 Agent 实现。

核心原理：
  如果两次请求的 prompt 从第一个 token 开始字节级完全一致，
  DeepSeek 服务端会复用已计算的 KV cache（缓存命中 token 价格仅为 10%）。

三条铁律：
  1. 三段式上下文：不可变前缀 + 仅追加日志（运行时思考不进 API）
  2. 字节稳定前缀：system prompt + tools 在整个会话生命周期字节级不变
  3. 仅追加原则：历史消息只能 append，不能修改/删除/重排
     （例外：归档时更新 system prompt 并清空日志，此为设计允许的唯一修改点）

依赖：Python 标准库（json, gzip, urllib, time, logging）— 零外部依赖！
"""
import json, gzip, time, logging, copy
import urllib.request

logger = logging.getLogger("cache_agent")

# ─── 数据结构 ────────────────────────────────────────────────

class CacheStats:
    """单次请求的缓存统计"""
    __slots__ = ("prompt_tokens", "cached_tokens", "completion_tokens",
                 "latency_ms")
    def __init__(self, prompt_tokens=0, cached_tokens=0,
                 completion_tokens=0, latency_ms=0.0):
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / max(1, self.prompt_tokens)

    @property
    def cost_usd(self) -> float:
        uncached = self.prompt_tokens - self.cached_tokens
        return (uncached * 0.0000005 + self.cached_tokens * 0.00000005
                + self.completion_tokens * 0.000002)


# ─── 确定性序列化 ───────────────────────────────────────────

def _stable_tools(tools: list) -> list:
    """按函数名排序工具列表"""
    return sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))


# ─── API 调用（标准库实现，零依赖） ──────────────────────────

_DEEPSEEK_API = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"

def _api_call(messages: list, tools: list, api_key: str,
              api_url: str = _DEEPSEEK_API, stream: bool = False,
              **kwargs) -> tuple[dict, float]:
    """
    调用 DeepSeek Chat API。
    返回 (response_dict, elapsed_ms)。
    使用标准库 urllib + gzip，零外部依赖。
    """
    body = {
        "model": kwargs.get("model", _DEEPSEEK_MODEL),
        "messages": messages,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    if "max_tokens" in kwargs:
        body["max_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        body["temperature"] = kwargs["temperature"]

    payload = json.dumps(body).encode("utf-8")
    should_gzip = len(payload) > 1024
    if should_gzip:
        payload = gzip.compress(payload)

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CacheAwareAgent/1.0",
        },
        method="POST",
    )
    if should_gzip:
        req.add_header("Content-Encoding", "gzip")

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except Exception as e:
        raise RuntimeError(f"API 调用失败: {e}") from e
    elapsed = (time.time() - start) * 1000

    result = json.loads(raw.decode("utf-8"))
    return result, elapsed


def _generate_summary(messages: list, api_key: str) -> str:
    """用 DeepSeek Chat 生成会话摘要"""
    if not messages:
        return ""
    body = {
        "model": _DEEPSEEK_MODEL,
        "messages": [
            {"role": "system",
             "content": "你是一个会话摘要生成器。"
                        "用3-5句话总结：目标、已做、待做、用户偏好。"},
            *messages[-50:],
        ],
        "max_tokens": 500,
        "temperature": 0.3,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _DEEPSEEK_API,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CacheAwareAgent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")) or ""
    except Exception:
        return ""


# ─── 主类 ───────────────────────────────────────────────────

class CacheAwareAgent:
    """
    缓存感知 Agent。

    用法:
        agent = CacheAwareAgent(
            system_prompt="你是一个AI助手",
            tools=[...],
            api_key="sk-xxx",
        )
        agent.add_user_message("今天修了什么bug？")
        reply = agent.send()
        print(reply)
        print(agent.report())  # 缓存命中报告
    """

    MAX_MESSAGES = 300
    MAX_ESTIMATED_TOKENS = 80000

    def __init__(
        self,
        system_prompt: str,
        tools: list = None,
        api_key: str = "",
        api_url: str = _DEEPSEEK_API,
    ):
        self._system_prompt = system_prompt.strip()
        self._tools_raw = _stable_tools(tools or [])
        self._api_key = api_key
        self._api_url = api_url
        self._log: list[dict] = []

        # 运行时暂存（绝不进入 API）
        self.scratchpad: dict = {}

        # 缓存统计
        self.stats: list[CacheStats] = []
        self._last_cache: CacheStats | None = None

        # 归档
        self._archived_count = 0

    # ── 消息操作（仅追加） ──────────────────────────────────

    def add_user_message(self, content: str):
        self._log.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str = "",
                              tool_calls: list = None):
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = sorted(tool_calls,
                                       key=lambda x: x.get("id", ""))
        self._log.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str):
        self._log.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(content),
        })

    # ── 请求构造 ─────────────────────────────────────────────

    def build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system_prompt},
                *self._log]

    def build_tools(self) -> list:
        return copy.deepcopy(self._tools_raw)

    # ── API 调用 ─────────────────────────────────────────────

    def send(self, **kwargs) -> str:
        """
        发送请求到 DeepSeek API。
        返回 assistant 的文本回复。
        异常直接抛出，不追加消息到日志。
        """
        result, elapsed = _api_call(
            messages=self.build_messages(),
            tools=self.build_tools(),
            api_key=self._api_key,
            api_url=self._api_url,
            **kwargs,
        )

        # 缓存统计
        usage = result.get("usage", {})
        if usage:
            cached = (usage.get("prompt_cache_hit_tokens", 0)
                      or usage.get("cached_tokens", 0))
            cs = CacheStats(
                prompt_tokens=usage.get("prompt_tokens", 0),
                cached_tokens=cached,
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=elapsed,
            )
            self.stats.append(cs)
            self._last_cache = cs

        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls")

        if tool_calls_raw:
            tc_dicts = [
                {"id": t.get("id"), "type": "function",
                 "function": t.get("function", {})}
                for t in tool_calls_raw
            ]
            self.add_assistant_message(content=content or "", tool_calls=tc_dicts)
        else:
            self.add_assistant_message(content=content or "")

        if self.should_archive():
            self._do_archive()
            logger.info("会话已自动归档 (#%d)", self._archived_count)

        return content or ""

    # ── 归档 ─────────────────────────────────────────────────

    def should_archive(self) -> bool:
        return (len(self._log) >= self.MAX_MESSAGES
                or self._estimate_tokens() >= self.MAX_ESTIMATED_TOKENS)

    def _estimate_tokens(self) -> int:
        total = len(self._system_prompt)
        for msg in self._log:
            total += len(json.dumps(msg, ensure_ascii=False))
        return total // 4

    def _do_archive(self):
        """
        归档：生成摘要 → 更新 system prompt → 清空日志。
        唯一允许修改 system prompt 和清空日志的地方。
        """
        try:
            summary = _generate_summary(self._log, self._api_key)
        except Exception as e:
            logger.warning("归档摘要生成失败: %s", e)
            self._log = self._log[-5:]  # 保底保留最近5条
            return

        self._archived_count += 1
        self._system_prompt += (
            "\n\n【会话历史摘要 #" + str(self._archived_count) + "】\n"
            + summary
        )
        self._log = []

    def create_successor(self):
        """归档后创建新 Agent 实例"""
        backup = list(self._log)
        try:
            summary = _generate_summary(self._log, self._api_key)
        except Exception:
            summary = ""
        finally:
            self._log = backup

        new_prompt = (self._system_prompt
                      + "\n\n【会话历史摘要】\n" + summary)
        succ = CacheAwareAgent(
            system_prompt=new_prompt,
            tools=self._tools_raw,
            api_key=self._api_key,
            api_url=self._api_url,
        )
        succ._archived_count = self._archived_count + 1
        return succ

    # ── 统计 ─────────────────────────────────────────────────

    @property
    def last_cache_stats(self) -> CacheStats | None:
        return self._last_cache

    def report(self) -> dict:
        if not self.stats:
            return {"status": "no_data"}
        total_prompt = sum(s.prompt_tokens for s in self.stats)
        total_cached = sum(s.cached_tokens for s in self.stats)
        total_cost = sum(s.cost_usd for s in self.stats)
        avg_lat = (sum(s.latency_ms for s in self.stats)
                   / len(self.stats))
        warmed = self.stats[1:] if len(self.stats) > 1 else []
        warm_hit = (sum(s.hit_rate for s in warmed)
                    / max(1, len(warmed)))
        return {
            "total_rounds": len(self.stats),
            "total_prompt_tokens": total_prompt,
            "total_cached_tokens": total_cached,
            "overall_hit_rate": total_cached / max(1, total_prompt),
            "warmed_hit_rate": round(warm_hit, 4),
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_lat, 1),
            "cost_saved_pct": (
                f"{total_cached / max(1, total_prompt) * 0.9 * 100:.1f}%"
            ),
            "archived_count": self._archived_count,
        }


# ─── 自测（纯逻辑，不调用 API） ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = CacheAwareAgent(
        system_prompt="你是测试助手。保持回复简短。",
        api_key="sk-placeholder",
    )

    # 1. 构造
    msgs = agent.build_messages()
    assert len(msgs) == 1 and msgs[0]["role"] == "system"
    print("[PASS] 构造: 1 system msg")

    # 2. 追加
    agent.add_user_message("你好")
    agent.add_assistant_message(content="你好！")
    assert len(agent._log) == 2
    print(f"[PASS] 追加: {len(agent._log)} msgs")

    # 3. 字节稳定性
    p1 = agent._system_prompt
    p2 = CacheAwareAgent(system_prompt=agent._system_prompt,
                          tools=agent._tools_raw,
                          api_key="k")._system_prompt
    assert p1 == p2
    print("[PASS] 字节稳定: prefix unchanged")

    # 4. tools 排序
    t1 = [{"function": {"name": "z_func"}, "type": "function"},
           {"function": {"name": "a_func"}, "type": "function"}]
    t2 = _stable_tools(t1)
    assert t2[0]["function"]["name"] == "a_func"
    print(f"[PASS] tools 排序: {[t['function']['name'] for t in t2]}")

    # 5. 归档不产生双 system
    agent._do_archive()
    msgs_after = agent.build_messages()
    sys_count = sum(1 for m in msgs_after if m["role"] == "system")
    assert sys_count == 1, f"预期1条system，实际{sys_count}条"
    print(f"[PASS] 归档: 1 system msg (无重复)")

    # 6. build_tools 返回副本
    agent2 = CacheAwareAgent(system_prompt="t", tools=t1, api_key="k")
    bt = agent2.build_tools()
    bt.append({"function": {"name": "evil"}, "type": "function"})
    assert len(agent2._tools_raw) == 2
    print("[PASS] build_tools 返回副本")

    # 7. create_successor 优雅降级
    agent3 = CacheAwareAgent(system_prompt="t", api_key="k")
    agent3.add_user_message("hi")
    succ = agent3.create_successor()
    assert succ is not None
    assert succ._archived_count == agent3._archived_count + 1
    print("[PASS] create_successor")

    # 8. 缓存统计
    cs = CacheStats(prompt_tokens=1000, cached_tokens=900,
                    completion_tokens=200, latency_ms=500)
    assert cs.hit_rate == 0.9
    assert cs.cost_usd > 0
    print(f"[PASS] 缓存统计: hit_rate={cs.hit_rate}, cost=${cs.cost_usd}")

    print(f"\n=== 全部 8 项自测通过 ===")
    print(f"零外部依赖：仅标准库")
