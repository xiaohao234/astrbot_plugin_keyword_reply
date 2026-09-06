# -*- coding: utf-8 -*-
"""AstrBot 关键词自动回复插件。

在 WebUI 中配置「关键词 -> 回复内容」规则：
- 关键词每行一个，命中任意一个即触发，支持在句子中间匹配；
- 回复内容支持多行，用单独一行 ``---`` 分隔多条回复，每条回复会作为一条独立消息依次发送；
- 回复内容中可用行首 ``[图片]``（或 ``[img]``）标记发送图片，后面跟图片 URL 或本地路径；
  标记行与普通文字行可任意混排，从而自由控制图片在回复序列中的位置
  （例如：先发一条文本消息，再发一张图片，最后再发一条文本消息）；
  同一块内的文字与图片会组装进同一条 MessageChain 一起发出；
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
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star

# 多条回复之间的分隔符（需要单独占一行）
REPLY_SEPARATOR = "---"

# 图片标记：出现在某行行首（忽略首尾空白与大小写）时，该行表示一张图片，
# 标记后的剩余内容为图片 URL（http/https）或本地文件路径
IMAGE_MARKERS = ("[图片]", "[img]")

# 发送间隔的上限（秒），防止配置成异常大的值
MAX_SEND_INTERVAL = 60.0

# 模块级运行时状态：用于在「自定义过滤器」与「消息处理器」之间共享已解析的配置。
# 该对象在插件每次加载 / 重载（含 WebUI 保存配置触发的热重载）时被整体重新填充，
# 其大小只取决于规则数量，不会随消息数量增长，因此不存在内存泄漏。
_runtime: dict[str, Any] = {
    "enabled": True,
    "case_sensitive": False,
    "send_interval": 0.5,
    # 编译后的规则：[{"keywords": [str, ...],
    #                "replies": [[{"type": "text", "text": str} | {"type": "image", "src": str}, ...], ...]}, ...]
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


def _extract_image_source(line: str) -> str | None:
    """判断一行是否为图片标记行：是则返回图片来源，否则返回 None。

    图片标记为行首（忽略首尾空白与大小写）的 ``[图片]`` 或 ``[img]``，
    标记后的剩余内容为图片 URL（http/https）或本地文件路径；
    返回空字符串表示该行是图片标记但未填写来源。
    """
    stripped = line.strip()
    lowered = stripped.lower()
    for marker in IMAGE_MARKERS:
        if lowered.startswith(marker):
            src = stripped[len(marker):].strip()
            if src.lower().startswith("file://"):
                src = src[len("file://"):]
            return src
    return None


def _parse_block_components(block_lines: list[str]) -> list[dict[str, str]]:
    """把单个回复块解析为按原文顺序排列的组件列表（图文混排）。

    普通文字行合并为一个文本组件（保留内部换行）；
    图片标记行解析为一个图片组件（type=image）。
    未填写来源的图片标记行记日志后跳过；全部为空时返回空列表。
    """
    components: list[dict[str, str]] = []
    text_lines: list[str] = []

    def flush_text() -> None:
        if not text_lines:
            return
        content = "\n".join(text_lines).strip()
        text_lines.clear()
        if content:
            components.append({"type": "text", "text": content})

    for line in block_lines:
        src = _extract_image_source(line)
        if src is None:
            text_lines.append(line)
            continue
        # 遇到图片标记：先收尾前面累计的文字，再记录图片
        flush_text()
        if src:
            components.append({"type": "image", "src": src})
        else:
            logger.warning("[keyword_reply] 回复内容中有图片标记但未填写来源，已跳过该图片")
    flush_text()
    return components


def _parse_replies(raw: str) -> list[list[dict[str, str]]]:
    """把回复内容解析为多条消息，每条消息是一组有序组件。

    规则：
    - 用单独一行 ``---`` 分隔多条回复，每条回复作为一条独立消息依次发送；
    - 单条回复内部可以多行，文字行与 ``[图片]`` 标记行可任意混排：
      文字行解析为文本组件，标记行解析为图片组件；
    - 因此可以自由控制图片的位置，例如：
      先发一条文本消息 -> 再发一张图片 -> 最后再发一条文本消息；
      同一块内的文字与图片会组装进同一条消息一起发出。
    """
    if raw is None:
        return []

    blocks: list[list[str]] = [[]]
    for line in str(raw).splitlines():
        if line.strip() == REPLY_SEPARATOR:
            blocks.append([])
        else:
            blocks[-1].append(line)

    replies: list[list[dict[str, str]]] = []
    for block in blocks:
        # 去掉块首尾的空白行
        while block and not block[0].strip():
            block.pop(0)
        while block and not block[-1].strip():
            block.pop()
        components = _parse_block_components(block)
        if components:
            replies.append(components)
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


def _build_message_chain(components: list[dict[str, str]]) -> MessageChain | None:
    """把编译后的组件列表构建为可发送的 MessageChain。

    每次发送都重新构建组件实例（不复用上次的组件对象），避免适配器
    对同一组件实例产生状态残留；构建结果为空时返回 None。
    """
    parts: list[Any] = []
    for comp in components:
        if comp["type"] == "text":
            parts.append(Plain(comp["text"]))
        elif comp["type"] == "image":
            src = comp["src"]
            if src.lower().startswith(("http://", "https://")):
                parts.append(Image.fromURL(src))
            else:
                parts.append(Image.fromFileSystem(src))
    return MessageChain(parts) if parts else None


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
        """检测到关键词后自动回复（可多条、多行、图文混排，依次发送）。"""
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
            for message in rule["replies"]:
                # 已发送过至少一条且配置了间隔时，等待后再发下一条，防止平台风控
                if sent > 0 and interval > 0:
                    await asyncio.sleep(interval)
                try:
                    chain = _build_message_chain(message)
                    if chain is None:
                        continue
                    await event.send(chain)
                    sent += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"[keyword_reply] 发送回复失败: {exc}")

        if sent > 0:
            # 命中关键词并成功回复后，阻断事件继续传播，避免 LLM 再回复一次
            event.stop_event()
