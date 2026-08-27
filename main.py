# -*- coding: utf-8 -*-
"""AstrBot 关键词自动回复插件。

在 WebUI 中配置「关键词 -> 回复内容」规则：
- 关键词每行一个，命中任意一个即触发，支持在句子中间匹配；
- 回复内容支持多行，用单独一行 ``---`` 分隔多条回复，每条回复会作为一条独立消息依次发送；
- 多条回复之间可配置发送间隔（默认 0.5 秒），防止平台风控。

实现说明：
- 使用自定义过滤器（CustomFilter）判断消息是否命中关键词，只有命中时才会唤醒插件，
  从而不会干扰机器人正常的 @ 唤醒 / LLM 对话流程；
- 命中后通过 ``event.send()`` 直接发送回复，并在发送成功后 ``event.stop_event()``
  阻断事件继续传播，避免 LLM 重复回复；
- 所有匹配逻辑均为纯函数，无后台任务、无外部资源、无全局累积状态，不存在内存泄漏。
"""

from __future__ import annotations

import asyncio
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import MessageChain, filter, AstrMessageEvent
from astrbot.api.event.filter import CustomFilter
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star

# 多条回复之间的分隔符（需要单独占一行）
REPLY_SEPARATOR = "---"

# 发送间隔的上限（秒），防止配置成异常大的值
MAX_SEND_INTERVAL = 60.0

# 模块级运行时状态：用于在「自定义过滤器」与「消息处理器」之间共享已解析的配置。
# 该对象在插件每次加载 / 重载（含 WebUI 保存配置触发的热重载）时被整体重新填充，
# 其大小只取决于规则数量，不会随消息数量增长，因此不存在内存泄漏。
_runtime: dict[str, Any] = {
    "enabled": True,
    "case_sensitive": False,
    "send_interval": 0.5,
    # 编译后的规则：[{"keywords": [str, ...], "replies": [str, ...]}, ...]
    "compiled_rules": [],
}


def _parse_keywords(raw: str) -> list[str]:
    """把多行文本解析为关键词列表（去空行、去首尾空白）。"""
    if not raw:
        return []
    keywords: list[str] = []
    for line in str(raw).splitlines():
        kw = line.strip()
        if kw:
            keywords.append(kw)
    return keywords


def _parse_replies(raw: str) -> list[str]:
    """把回复内容解析为多条回复。

    规则：
    - 每条回复内部可以多行；
    - 用单独一行 ``---`` 分隔多条回复；
    - 每条回复会作为一条独立消息依次发送。
    """
    if raw is None:
        return []

    blocks: list[list[str]] = [[]]
    for line in str(raw).splitlines():
        if line.strip() == REPLY_SEPARATOR:
            blocks.append([])
        else:
            blocks[-1].append(line)

    replies: list[str] = []
    for block in blocks:
        # 去掉块首尾的空白行
        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()
        content = "\n".join(block).strip()
        if content:
            replies.append(content)
    return replies


def _coerce_interval(value: Any) -> float:
    """把配置值安全转换为 [0, 60] 区间内的发送间隔（秒）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    if v > MAX_SEND_INTERVAL:
        return MAX_SEND_INTERVAL
    return v


def _compile_rules(raw_rules: Any) -> list[dict[str, Any]]:
    """把配置中的规则列表编译为便于匹配的结构。

    跳过关键词或回复内容为空的规则（此类规则不会产生任何实际效果）。
    """
    if not isinstance(raw_rules, list):
        return []
    compiled: list[dict[str, Any]] = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            continue
        keywords = _parse_keywords(rule.get("keywords", ""))
        replies = _parse_replies(rule.get("reply", ""))
        if not keywords or not replies:
            continue
        compiled.append({"keywords": keywords, "replies": replies})
    return compiled


def _match_rules(
    text: str,
    compiled: list[dict[str, Any]],
    case_sensitive: bool,
) -> list[dict[str, Any]]:
    """返回命中的规则列表（保持配置顺序）。子串匹配。"""
    if not text:
        return []
    haystack = text if case_sensitive else text.lower()
    matched: list[dict[str, Any]] = []
    for rule in compiled:
        hit = False
        for kw in rule["keywords"]:
            needle = kw if case_sensitive else kw.lower()
            if needle in haystack:
                hit = True
                break
        if hit:
            matched.append(rule)
    return matched


class KeywordReplyFilter(CustomFilter):
    """仅当消息命中任一「带有效回复的关键词」时才唤醒插件。

    这样不会影响机器人正常的 @ 唤醒 / LLM 对话流程。
    """

    def __init__(self, raise_error: bool = False):
        super().__init__(raise_error=raise_error)

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        try:
            if not _runtime["enabled"]:
                return False
            text = (event.get_message_str() or "").strip()
            if not text:
                return False
            return bool(
                _match_rules(
                    text,
                    _runtime["compiled_rules"],
                    _runtime["case_sensitive"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[keyword_reply] 过滤器执行异常: {exc}")
            return False


class KeywordReplyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        # config 在插件声明了 _conf_schema.json 时始终由 AstrBot 注入；
        # 此处兜底为普通 dict，避免极端情况下的 None 访问。
        self.config = config if config is not None else {}
        self._sync_runtime()

    def _sync_runtime(self) -> None:
        """把当前配置同步到模块级运行时状态。"""
        cfg = self.config
        _runtime["enabled"] = bool(cfg.get("enabled", True))
        _runtime["case_sensitive"] = bool(cfg.get("case_sensitive", False))
        _runtime["send_interval"] = _coerce_interval(cfg.get("send_interval", 0.5))
        _runtime["compiled_rules"] = _compile_rules(cfg.get("rules"))

    async def initialize(self):
        self._sync_runtime()
        logger.info(
            f"[keyword_reply] 已加载 {len(_runtime['compiled_rules'])} 条规则"
            f"（启用={_runtime['enabled']}，区分大小写={_runtime['case_sensitive']}，"
            f"发送间隔={_runtime['send_interval']}s）"
        )

    async def terminate(self):
        # 无后台任务、无外部资源，无需清理
        pass

    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.custom_filter(KeywordReplyFilter, False)
    async def on_keyword_message(self, event: AstrMessageEvent):
        """检测到关键词后自动回复（可多条、多行、依次发送）。"""
        if not _runtime["enabled"]:
            return

        text = (event.get_message_str() or "").strip()
        matched = _match_rules(
            text,
            _runtime["compiled_rules"],
            _runtime["case_sensitive"],
        )
        if not matched:
            return

        sent = 0
        interval = _runtime["send_interval"]
        for rule in matched:
            for reply in rule["replies"]:
                # 已发送过至少一条且配置了间隔时，等待后再发下一条，防止平台风控
                if sent > 0 and interval > 0:
                    await asyncio.sleep(interval)
                try:
                    await event.send(MessageChain([Plain(reply)]))
                    sent += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[keyword_reply] 发送回复失败: {exc}")

        if sent > 0:
            # 命中关键词并成功回复后，阻断事件继续传播，避免 LLM 再回复一次
            event.stop_event()
