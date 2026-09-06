# -*- coding: utf-8 -*-
"""AstrBot 关键词自动回复插件。

在 WebUI 中配置「关键词 -> 回复内容」规则：
- 关键词每行一个，命中任意一个即触发，支持在句子中间匹配；
- 可开启「整词匹配」：含英文/数字的关键词必须独立出现才触发
  （如关键词 mj 不再命中 mmj、mja2），中文关键词不受影响；
- 回复内容支持多行，用单独一行 ``---`` 分隔多条回复，每条回复会作为一条独立消息依次发送；
- 回复内容中可用行首 ``[图片]``（或 ``[img]``）标记发送图片，后面跟图片来源，
  标记行与普通文字行可任意混排，从而自由控制图片在回复序列中的位置
  （例如：先发一条文本消息，再发一张图片，最后再发一条文本消息）；
  同一块内的文字与图片会组装进同一条 MessageChain 一起发出；
- 图片来源支持三种写法：网络 URL（http/https）、本地路径，以及引用规则「图片池」：
  在 WebUI 上传图片到规则的 images 字段后，用 ``[图片:编号]``（如 ``[图片:1]``）
  或 ``[图片:文件名]``（如 ``[图片:欢迎.gif]``）引用，上传的 GIF 动图等格式原样透传；
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
import os
import re
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import MessageChain, filter, AstrMessageEvent
from astrbot.api.event.filter import CustomFilter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools

# 插件名（与 metadata.yaml 的 name 一致），用于定位插件数据目录
PLUGIN_NAME = "astrbot_plugin_keyword_reply"

# 多条回复之间的分隔符（需要单独占一行）
REPLY_SEPARATOR = "---"

# 图片标记：出现在某行行首（忽略首尾空白与大小写）时，该行表示一张图片。
# 两种写法：``[图片]来源``（来源跟在标记后）或 ``[图片:来源]``（来源写在括号内）。
# 来源可以是 URL、本地路径，或规则图片池的编号/文件名引用。
IMAGE_MARKERS = ("[图片]", "[img]")

# 发送间隔的上限（秒），防止配置成异常大的值
MAX_SEND_INTERVAL = 60.0

# 模块级运行时状态：用于在「自定义过滤器」与「消息处理器」之间共享已解析的配置。
# 该对象在插件每次加载 / 重载（含 WebUI 保存配置触发的热重载）时被整体重新填充，
# 其大小只取决于规则数量，不会随消息数量增长，因此不存在内存泄漏。
_runtime: dict[str, Any] = {
    "enabled": True,
    "case_sensitive": False,
    "whole_word": False,
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


def _plugin_data_dir() -> str:
    """返回插件数据目录（data/plugin_data/<插件名>）。

    优先通过 StarTools 获取；不可用时回退为相对进程工作目录的路径。
    """
    try:
        return str(StarTools.get_data_dir(PLUGIN_NAME))
    except Exception:  # noqa: BLE001
        return os.path.join("data", "plugin_data", PLUGIN_NAME)


def _resolve_local_path(src: str) -> str:
    """把本地路径解析为可发送的绝对路径。

    相对路径优先按「插件数据目录」解析（WebUI 上传的文件存在那里），
    文件确实存在时才采用；否则回退为按进程工作目录解析的绝对路径。
    """
    if os.path.isabs(src):
        return src
    candidate = os.path.join(_plugin_data_dir(), src)
    if os.path.isfile(candidate):
        return candidate
    return os.path.abspath(src)


def _resolve_image_source(src: str, image_pool: list[str]) -> str | None:
    """把标记后的图片来源解析为可直接发送的 URL / 绝对路径。

    解析优先级：
    - ``http://`` / ``https://`` 开头 → 网络图片 URL；
    - 纯数字 → 引用规则图片池中第 N 张（从 1 开始）；
    - 不含路径分隔符 → 在规则图片池中按文件名（basename）匹配；
    - 其余 → 本地路径（相对路径按插件数据目录解析）。
    无法解析时记录日志并返回 None（该图片会被跳过）。
    """
    lowered = src.lower()
    if lowered.startswith(("http://", "https://")):
        return src
    if lowered.startswith("file://"):
        return _resolve_local_path(src[len("file://"):])
    if src.isdigit():
        index = int(src)
        if 1 <= index <= len(image_pool):
            return image_pool[index - 1]
        logger.warning(
            f"[keyword_reply] [图片:{src}] 超出规则图片池范围（共 {len(image_pool)} 张），已跳过"
        )
        return None
    if "/" not in src and "\\" not in src:
        name = src.lower()
        for path in image_pool:
            if os.path.basename(path).lower() == name:
                return path
        logger.warning(f"[keyword_reply] 未在规则图片池中找到图片「{src}」，已跳过")
        return None
    return _resolve_local_path(src)


def _compile_image_pool(raw_images: Any) -> list[str]:
    """把规则的 images 字段（WebUI 上传生成的路径列表）编译为可发送的图片池。

    本地路径在编译期解析为绝对路径（相对路径按插件数据目录解析），
    文件不存在的条目记日志后忽略，保证「[图片:编号]」引用始终指向有效文件；
    URL 条目原样保留。
    """
    if not isinstance(raw_images, list):
        return []
    pool: list[str] = []
    for item in raw_images:
        if not isinstance(item, str) or not item.strip():
            continue
        src = item.strip()
        if src.lower().startswith(("http://", "https://")):
            pool.append(src)
            continue
        path = _resolve_local_path(src)
        if os.path.isfile(path):
            pool.append(path)
        else:
            logger.warning(f"[keyword_reply] 规则图片池中的文件不存在，已忽略: {src}")
    return pool


def _extract_image_source(line: str) -> str | None:
    """判断一行是否为图片标记行：是则返回标记后的原始来源，否则返回 None。

    支持两种标记写法（均忽略首尾空白与大小写）：
    - ``[图片]来源`` / ``[img]来源``：标记后整行剩余内容为来源；
    - ``[图片:来源]`` / ``[img:来源]``：来源写在方括号内（适合引用图片池）。
    返回空字符串表示该行是图片标记但未填写来源。
    """
    stripped = line.strip()
    colon_form = re.match(r"^\[(?:图片|img):([^\]]*)\]$", stripped, re.IGNORECASE)
    if colon_form:
        return colon_form.group(1).strip()
    lowered = stripped.lower()
    for marker in IMAGE_MARKERS:
        if lowered.startswith(marker):
            return stripped[len(marker):].strip()
    return None


def _parse_block_components(
    block_lines: list[str],
    image_pool: list[str],
) -> list[dict[str, str]]:
    """把单个回复块解析为按原文顺序排列的组件列表（图文混排）。

    普通文字行合并为一个文本组件（保留内部换行）；
    图片标记行解析为一个图片组件（type=image），来源中的图片池引用
    会在编译期解析为绝对路径。无法解析的图片记日志后跳过；
    全部为空时返回空列表。
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
        if not src:
            logger.warning("[keyword_reply] 回复内容中有图片标记但未填写来源，已跳过该图片")
            continue
        resolved = _resolve_image_source(src, image_pool)
        if resolved:
            components.append({"type": "image", "src": resolved})
    flush_text()
    return components


def _parse_replies(
    raw: str,
    image_pool: list[str] | None = None,
) -> list[list[dict[str, str]]]:
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
    pool = image_pool or []

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
        components = _parse_block_components(block, pool)
        if components:
            replies.append(components)
    return replies


def _coerce_interval(value: Any) -> float:
    """把配置值安全转换为 [0, 60] 区间内的发送间隔（秒）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        # NaN：与任何值比较均为 False，会绕过下方上下限收敛；
        # 一旦进入 asyncio.sleep 会让回复序列在空闲时停住，直接取默认值
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
        image_pool = _compile_image_pool(rule.get("images"))
        replies = _parse_replies(rule.get("reply", ""), image_pool)
        if not keywords or not replies:
            continue
        compiled.append({"keywords": keywords, "replies": replies})
    return compiled


def _is_ascii_word_char(ch: str) -> bool:
    """判断字符是否为「英文粘连字符」：ASCII 字母 / 数字 / 下划线。

    整词匹配只把这类字符视为粘连，中文等全角字符不算，
    因此「这是mj」「mj吧」这类中英混排场景不受整词模式影响。
    """
    return ch.isascii() and (ch.isalnum() or ch == "_")


def _keyword_hit(haystack: str, needle: str, whole_word: bool) -> bool:
    """判断单个关键词是否命中（调用方已处理大小写归一）。

    - 整词模式关闭：子串匹配（原行为）；
    - 整词模式开启：关键词必须独立出现——命中位置的前一个字符和
      后一个字符都不能是 ASCII 字母 / 数字 / 下划线（如关键词 mj
      不再命中 mmj、mja2），全部位置都粘连时视为未命中。
    """
    if not whole_word:
        return needle in haystack

    start = haystack.find(needle)
    while start != -1:
        before_ok = start == 0 or not _is_ascii_word_char(haystack[start - 1])
        end = start + len(needle)
        after_ok = end >= len(haystack) or not _is_ascii_word_char(haystack[end])
        if before_ok and after_ok:
            return True
        start = haystack.find(needle, start + 1)
    return False


def _match_rules(
    text: str,
    compiled: list[dict[str, Any]],
    case_sensitive: bool,
    whole_word: bool = False,
) -> list[dict[str, Any]]:
    """返回命中的规则列表（保持配置顺序）。

    整词模式开启时，英文/数字关键词必须独立出现才命中，
    中文关键词不受影响（仍支持句子中间匹配）。
    """
    if not text:
        return []
    haystack = text if case_sensitive else text.lower()
    matched: list[dict[str, Any]] = []
    for rule in compiled:
        hit = False
        for kw in rule["keywords"]:
            needle = kw if case_sensitive else kw.lower()
            if _keyword_hit(haystack, needle, whole_word):
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
                    _runtime["whole_word"],
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
        _runtime["whole_word"] = bool(cfg.get("whole_word", False))
        _runtime["send_interval"] = _coerce_interval(cfg.get("send_interval", 0.5))
        _runtime["compiled_rules"] = _compile_rules(cfg.get("rules"))

    async def initialize(self):
        self._sync_runtime()
        logger.info(
            f"[keyword_reply] 已加载 {len(_runtime['compiled_rules'])} 条规则"
            f"（启用={_runtime['enabled']}，区分大小写={_runtime['case_sensitive']}，"
            f"整词匹配={_runtime['whole_word']}，"
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
            _runtime["whole_word"],
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
