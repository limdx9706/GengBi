"""续写输出栏：思维链查看 + 正则开关测试。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from novelforge.core.config import get_default_config
from novelforge.core.post_processor import (
    compose_thinking_display,
    extract_thinking_content,
    post_process_content,
)
from novelforge.core.regex_engine import RegexEngine
from novelforge.models.chapter import Continuation
from novelforge.models.regex import PLACEMENT_AI_OUTPUT, RegexScript


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


def test_default_config_output_prefs() -> None:
    cont = get_default_config()["continuation"]
    assert cont.get("show_thinking") is False
    assert cont.get("apply_regex") is True


def test_extract_thinking_content() -> None:
    text = (
        "<novelforge_thinking>\n分析 A\n\n分析 B\n</novelforge_thinking>\n"
        "<novelforge_chapter>正文</novelforge_chapter>"
    )
    assert extract_thinking_content(text) == "分析 A\n\n分析 B"
    assert extract_thinking_content("无标签正文") == ""
    assert extract_thinking_content("") == ""
    multi = (
        "<novelforge_thinking>A</novelforge_thinking>\n"
        "中间\n"
        "<novelforge_thinking>B</novelforge_thinking>"
    )
    assert extract_thinking_content(multi) == "A\n\nB"


def test_compose_thinking_display() -> None:
    assert compose_thinking_display(None, None) == "当前段无思维链内容"
    out = compose_thinking_display("推理段", "标签段")
    assert "【模型思维链】" in out
    assert "【预设思维链】" in out
    assert "推理段" in out
    assert "标签段" in out


def test_compose_thinking_display_native_think_fallback() -> None:
    raw = "<think>原生思考</think>\n正文"
    out = compose_thinking_display("", "", raw_text=raw)
    assert "原生思考" in out
    assert "【模型思维链】" in out


def test_extract_native_think_content() -> None:
    from novelforge.core.post_processor import extract_native_think_content

    assert extract_native_think_content("<think>A</think>正文") == "A"
    assert extract_native_think_content("无标签") == ""


def test_llm_extract_reasoning_from_delta_fields() -> None:
    from novelforge.services.llm_client import LLMClient

    assert LLMClient._extract_reasoning_text({"reasoning_content": "ds"}) == "ds"
    assert LLMClient._extract_reasoning_text({"reasoning": "or"}) == "or"
    assert LLMClient._extract_reasoning_text({"thinking": "tk"}) == "tk"
    assert LLMClient._extract_reasoning_text({"reasoning": {"content": "嵌套"}}) == "嵌套"
    assert LLMClient.extract_message_reasoning({"reasoning": "msg"}) == "msg"


def test_parse_sse_reasoning_field_variants() -> None:
    from novelforge.services.llm_client import LLMClient

    client = LLMClient("https://api.example.com/v1", "sk-test")
    chunk = client._parse_sse_event(
        'data: {"choices":[{"delta":{"reasoning":"思考增量"}}]}'
    )
    assert chunk is not None
    assert chunk.reasoning_content == "思考增量"
    assert chunk.content == ""

    chunk2 = client._parse_sse_event(
        'data: {"choices":[{"delta":{"reasoning_content":"ds增量","content":"字"}}]}'
    )
    assert chunk2 is not None
    assert chunk2.reasoning_content == "ds增量"
    assert chunk2.content == "字"


def test_post_process_apply_regex_false_keeps_thinking_tag() -> None:
    engine = RegexEngine()
    hide = RegexScript(
        id="test_hide",
        scriptName="隐藏思维链",
        findRegex=r"/<novelforge_thinking>[\s\S]*?<\/novelforge_thinking>\n*/g",
        replaceString="",
        placement=[PLACEMENT_AI_OUTPUT],
    )
    engine.compile_scripts([hide])
    raw = "<novelforge_thinking>内部分析</novelforge_thinking>\n正文"
    skipped = post_process_content(raw, regex_engine=engine, apply_regex=False)
    assert "内部分析" in skipped
    assert "正文" in skipped
    applied = post_process_content(raw, regex_engine=engine, apply_regex=True)
    assert "内部分析" not in applied
    assert "正文" in applied


def test_post_process_always_strips_native_think() -> None:
    from novelforge.core.post_processor import strip_native_think_blocks

    raw = "<think>原生思考</think>\n正文段落"
    stripped = strip_native_think_blocks(raw)
    assert "原生思考" not in stripped
    assert "正文段落" in stripped
    processed = post_process_content(raw, regex_engine=None, apply_regex=False)
    assert "原生思考" not in processed
    assert "正文段落" in processed


def test_continuation_raw_content_default_and_roundtrip() -> None:
    cont = Continuation(id="sw_1", content="处理后", raw_content="处理前")
    assert cont.raw_content == "处理前"
    restored = Continuation.model_validate(cont.model_dump(mode="json"))
    assert restored.raw_content == "处理前"
    assert restored.content == "处理后"


def test_continuation_panel_output_prefs(qapp: QApplication) -> None:
    from novelforge.ui.continuation_panel import ContinuationPanel

    panel = ContinuationPanel()
    assert panel.thinking_edit.isVisible() is False
    assert panel._apply_regex_check.isChecked() is True
    assert panel._show_thinking_check.isChecked() is False

    panel.set_output_prefs(show_thinking=True, apply_regex=False)
    assert panel._show_thinking_check.isChecked() is True
    assert panel._apply_regex_check.isChecked() is False
    assert panel.thinking_edit.isVisible() is True

    params = panel.get_parameters()
    assert params["apply_regex"] is False


def test_legacy_swipe_display_does_not_reprocess(qapp: QApplication) -> None:
    """无 raw_content 的旧 swipe 不得把已处理后正文再跑一遍正则。"""
    from novelforge.ui.continuation_panel import ContinuationPanel

    engine = RegexEngine()
    mangler = RegexScript(
        id="test_mangle",
        scriptName="二次处理探针",
        findRegex=r"/处理后/g",
        replaceString="被二次处理",
        placement=[PLACEMENT_AI_OUTPUT],
    )
    engine.compile_scripts([mangler])
    panel = ContinuationPanel()
    panel.set_output_process_context(engine, None)
    panel.set_output_prefs(show_thinking=False, apply_regex=True)
    swipe = Continuation(id="sw_old", content="处理后正文", raw_content="")
    panel.set_current_swipe(swipe, [swipe])
    assert panel.output_edit.toPlainText() == "处理后正文"


def test_regex_toggle_uses_stored_or_raw(qapp: QApplication) -> None:
    """有原文时：开正则显示入库 content，关正则显示 raw_content。"""
    from novelforge.ui.continuation_panel import ContinuationPanel

    panel = ContinuationPanel()
    swipe = Continuation(id="sw_1", content="处理后", raw_content="原文标签")
    panel.set_output_prefs(show_thinking=False, apply_regex=True)
    panel.set_current_swipe(swipe, [swipe])
    assert panel.output_edit.toPlainText() == "处理后"
    panel.set_output_prefs(show_thinking=False, apply_regex=False)
    assert panel.output_edit.toPlainText() == "原文标签"
