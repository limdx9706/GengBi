"""思考强度（reasoning_effort）注入与 Qwen thinking_budget 清洗测试。

覆盖：
- LLMClient 的 reasoning_effort 属性存储与 payload 注入条件
- MainWindow._resolve_reasoning_effort 仅读端点配置
- Qwen 模型 thinking_budget 清洗（flash/max/extra_payload）
"""
from __future__ import annotations

from novelforge.models.preset import WritingPreset
from novelforge.services.llm_client import LLMClient, QWEN_THINKING_BUDGET_MAX


def _apply_extensions(
    model: str,
    *,
    reasoning_effort: str = "",
    extra_payload: dict | None = None,
) -> dict:
    """构建最小 payload 并走 _apply_payload_extensions。"""
    client = LLMClient(
        "https://api.example.com/v1",
        "sk-test",
        reasoning_effort=reasoning_effort or None,
        extra_payload=extra_payload,
    )
    payload: dict = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    client._apply_payload_extensions(payload, model)
    return payload


# ===== LLMClient reasoning_effort 属性与注入条件 =====


def test_llm_client_stores_reasoning_effort() -> None:
    """LLMClient 正确存储 reasoning_effort 参数。"""
    client = LLMClient("https://api.example.com/v1", "sk-test", reasoning_effort="high")
    assert client.reasoning_effort == "high"


def test_llm_client_stores_empty_reasoning_effort() -> None:
    """reasoning_effort 为 None 时存储为空串。"""
    client = LLMClient("https://api.example.com/v1", "sk-test")
    assert client.reasoning_effort == ""


def test_llm_client_payload_injects_reasoning_effort() -> None:
    """reasoning_effort="high" 时满足注入条件。"""
    client = LLMClient("https://api.example.com/v1", "sk-test", reasoning_effort="high")
    # 镜像 llm_client.py 中的注入条件
    should_inject = (
        client.reasoning_effort
        and client.reasoning_effort.lower() not in {"", "none", "off"}
    )
    assert should_inject is True
    assert client.reasoning_effort == "high"


def test_llm_client_payload_skips_empty_reasoning_effort() -> None:
    """空串不满足注入条件。"""
    client = LLMClient("https://api.example.com/v1", "sk-test", reasoning_effort="")
    should_inject = (
        client.reasoning_effort
        and client.reasoning_effort.lower() not in {"", "none", "off"}
    )
    assert not should_inject


def test_llm_client_payload_skips_none_off() -> None:
    """none/off（大小写不敏感）不满足注入条件。"""
    for val in ("none", "off", "None", "OFF", "None", "off"):
        client = LLMClient("https://api.example.com/v1", "sk-test", reasoning_effort=val)
        should_inject = (
            client.reasoning_effort
            and client.reasoning_effort.lower() not in {"", "none", "off"}
        )
        assert not should_inject, f"值 {val!r} 不应注入"


def test_llm_client_payload_injects_max() -> None:
    """max（DeepSeek V4）满足注入条件。"""
    client = LLMClient("https://api.example.com/v1", "sk-test", reasoning_effort="max")
    should_inject = (
        client.reasoning_effort
        and client.reasoning_effort.lower() not in {"", "none", "off"}
    )
    assert should_inject is True


# ===== _resolve_reasoning_effort 解析逻辑 =====


def _make_main_window_stub():
    """创建仅含 _resolve_reasoning_effort 方法的 MainWindow 桩实例。"""
    from novelforge.ui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    return win


def test_resolve_reasoning_effort_ignores_preset() -> None:
    """预设 reasoning_effort 不再参与解析，仅读端点。"""
    win = _make_main_window_stub()
    preset = WritingPreset(
        id="test-preset",
        name="test",
        generation_params={"reasoning_effort": "max"},
    )
    endpoint = {"reasoning_effort": "low"}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == "low"
    _ = preset  # 预设字段存在但不被读取


def test_resolve_reasoning_effort_preset_high_endpoint_empty() -> None:
    """预设 high、端点空时返回空串（不发送）——修复默认预设带崩 flash。"""
    win = _make_main_window_stub()
    _preset = WritingPreset(
        id="test-preset",
        name="test",
        generation_params={"reasoning_effort": "high"},
    )
    endpoint: dict = {"reasoning_effort": ""}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == ""


def test_resolve_reasoning_effort_endpoint_value() -> None:
    """端点 reasoning_effort 直接返回。"""
    win = _make_main_window_stub()
    endpoint = {"reasoning_effort": "medium"}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == "medium"


def test_resolve_reasoning_effort_no_preset() -> None:
    """端点配置直接返回。"""
    win = _make_main_window_stub()
    endpoint = {"reasoning_effort": "low"}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == "low"


def test_resolve_reasoning_effort_empty() -> None:
    """端点为空时返回空串（不发送）。"""
    win = _make_main_window_stub()
    endpoint = {"reasoning_effort": ""}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == ""


def test_resolve_reasoning_effort_endpoint_missing_field() -> None:
    """端点无 reasoning_effort 字段时返回空串。"""
    win = _make_main_window_stub()
    endpoint: dict = {}
    result = win._resolve_reasoning_effort(endpoint)
    assert result == ""


# ===== _ensure_user_message Gemini 兼容兜底 =====


def test_ensure_user_message_all_system_converts_last() -> None:
    """全 system 消息时，最后一条转为 user 角色。"""
    messages = [
        {"role": "system", "content": "破限文本"},
        {"role": "system", "content": "任务指令"},
    ]
    result = LLMClient._ensure_user_message(messages)
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == "任务指令"
    # 前面的消息保持 system
    assert result[0]["role"] == "system"
    # 原始列表不被修改
    assert messages[-1]["role"] == "system"


def test_ensure_user_message_has_user_unchanged() -> None:
    """已含 user 消息时不修改。"""
    messages = [
        {"role": "system", "content": "破限"},
        {"role": "user", "content": "任务"},
    ]
    result = LLMClient._ensure_user_message(messages)
    assert result == messages


def test_ensure_user_message_single_system_converts() -> None:
    """单条 system 消息也转为 user。"""
    messages = [{"role": "system", "content": "任务指令"}]
    result = LLMClient._ensure_user_message(messages)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "任务指令"


# ===== _is_xai_model Grok 模型检测 =====


def test_is_xai_model_grok() -> None:
    """检测 Grok 模型。"""
    assert LLMClient._is_xai_model("grok-4.5") is True
    assert LLMClient._is_xai_model("Grok-3") is True
    assert LLMClient._is_xai_model("grok-code-fast") is True
    assert LLMClient._is_xai_model("gpt-4") is False
    assert LLMClient._is_xai_model("gemini-pro") is False


# ===== _filter_unsupported_params 按模型过滤不支持参数 =====


def test_filter_unsupported_params_grok4_removes_penalties_and_reasoning() -> None:
    """Grok-4/4.5 删除 penalties 和 reasoning_effort。"""
    payload = {
        "model": "grok-4.5",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "reasoning_effort": "medium",
    }
    LLMClient._filter_unsupported_params(payload, "grok-4.5")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "reasoning_effort" not in payload


def test_filter_unsupported_params_grok3_mini_removes_penalties_keeps_reasoning() -> None:
    """Grok-3-mini 删除 penalties 但保留 reasoning_effort。"""
    payload = {
        "model": "grok-3-mini",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "reasoning_effort": "medium",
    }
    LLMClient._filter_unsupported_params(payload, "grok-3-mini")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "reasoning_effort" in payload


def test_filter_unsupported_params_grok_code_removes_penalties_and_reasoning() -> None:
    """grok-code 删除 penalties 和 reasoning_effort。"""
    payload = {
        "model": "grok-code-fast",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "reasoning_effort": "medium",
    }
    LLMClient._filter_unsupported_params(payload, "grok-code-fast")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "reasoning_effort" not in payload


def test_filter_unsupported_params_grok3_removes_penalties_and_reasoning() -> None:
    """grok-3（非 mini）删除 penalties 和 reasoning_effort。"""
    payload = {
        "model": "grok-3",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "reasoning_effort": "medium",
    }
    LLMClient._filter_unsupported_params(payload, "grok-3")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert "reasoning_effort" not in payload


def test_filter_unsupported_params_non_grok_unchanged() -> None:
    """非 Grok/Gemini 模型：非零 penalty 保留。"""
    payload = {
        "model": "gpt-4",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "reasoning_effort": "medium",
    }
    LLMClient._filter_unsupported_params(payload, "gpt-4")
    assert payload["presence_penalty"] == 0.5
    assert payload["frequency_penalty"] == 0.3
    assert payload["reasoning_effort"] == "medium"


def test_filter_unsupported_params_gemini_removes_penalties() -> None:
    """Gemini 模型删除 presence/frequency_penalty（含非零），避免 400 INVALID_ARGUMENT。"""
    payload = {
        "model": "gemini-2.5-pro",
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "temperature": 0.8,
    }
    LLMClient._filter_unsupported_params(payload, "gemini-2.5-pro")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert payload["temperature"] == 0.8


def test_filter_unsupported_params_gemini_zero_penalties_removed() -> None:
    """Gemini 即使 penalty=0.0 也不发送。"""
    payload = {
        "model": "google/gemini-3-flash",
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    LLMClient._filter_unsupported_params(payload, "google/gemini-3-flash")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload


def test_filter_unsupported_params_other_model_zero_penalties_omitted() -> None:
    """其他模型：penalty 为 0.0 时省略（与默认等价，更兼容）。"""
    payload = {
        "model": "gpt-4o",
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "temperature": 0.7,
    }
    LLMClient._filter_unsupported_params(payload, "gpt-4o")
    assert "presence_penalty" not in payload
    assert "frequency_penalty" not in payload
    assert payload["temperature"] == 0.7


# ===== Claude 采样参数弃用（Fable/Mythos/Opus 4.7+）=====


def test_is_sampling_deprecated_claude_model_hits() -> None:
    """Fable/Mythos/Opus 4.7+ 判定为采样参数已弃用。"""
    assert LLMClient._is_sampling_deprecated_claude_model("fable5") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude-fable-5") is True
    assert LLMClient._is_sampling_deprecated_claude_model("anthropic.claude-fable-5") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude-mythos-5") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude-opus-4-7") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude-opus-4.8") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude-opus-5") is True
    assert LLMClient._is_sampling_deprecated_claude_model("claude_opus_4_7") is True


def test_is_sampling_deprecated_claude_model_misses() -> None:
    """旧 Claude / 非 Claude 不应判定为采样弃用。"""
    assert LLMClient._is_sampling_deprecated_claude_model("claude-3-5-sonnet") is False
    assert LLMClient._is_sampling_deprecated_claude_model("claude-opus-4-6") is False
    assert LLMClient._is_sampling_deprecated_claude_model("gpt-4o") is False
    assert LLMClient._is_sampling_deprecated_claude_model("") is False


def test_filter_unsupported_params_claude_fable_removes_sampling() -> None:
    """fable5 / claude-fable-5 删除 temperature/top_p/top_k，并省略 0.0 penalty。"""
    for model in ("fable5", "claude-fable-5", "anthropic.claude-fable-5"):
        payload = {
            "model": model,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 40,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 2000,
        }
        LLMClient._filter_unsupported_params(payload, model)
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload
        assert "presence_penalty" not in payload
        assert "frequency_penalty" not in payload
        assert payload["max_tokens"] == 2000


def test_filter_unsupported_params_claude_opus47_and_opus5_removes_sampling() -> None:
    """claude-opus-4-7 / claude-opus-5 删除采样参数。"""
    for model in ("claude-opus-4-7", "claude-opus-5"):
        payload = {
            "model": model,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 10,
        }
        LLMClient._filter_unsupported_params(payload, model)
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload


def test_filter_unsupported_params_legacy_claude_keeps_sampling() -> None:
    """claude-3-5-sonnet / opus-4-6 / gpt-4o 保留采样参数。"""
    for model in ("claude-3-5-sonnet", "claude-opus-4-6", "gpt-4o"):
        payload = {
            "model": model,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.5,
        }
        LLMClient._filter_unsupported_params(payload, model)
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.5


# ===== Qwen thinking_budget 清洗 =====


def test_qwen38_flash_high_no_reasoning_effort_or_invalid_budget() -> None:
    """qwen3.8-flash + high：不发 reasoning_effort，不发非法 thinking_budget。"""
    for model in ("qwen3.8-flash", "qwen3.8flash", "Qwen/Qwen3.8-Flash"):
        payload = _apply_extensions(model, reasoning_effort="high")
        assert "reasoning_effort" not in payload
        assert "thinking_budget" not in payload


def test_qwen38_flash_low_maps_to_budget() -> None:
    """qwen3.8-flash + low → thinking_budget=4096。"""
    payload = _apply_extensions("qwen3.8-flash", reasoning_effort="low")
    assert "reasoning_effort" not in payload
    assert payload["thinking_budget"] == 4096


def test_qwen38_flash_empty_effort_uses_model_default() -> None:
    """端点未配置思考强度时，不注入 enable_thinking=False，走模型默认。"""
    payload = _apply_extensions("qwen3.8-flash", reasoning_effort="")
    assert "reasoning_effort" not in payload
    assert "thinking_budget" not in payload
    assert "enable_thinking" not in payload


def test_qwen38_flash_none_disables_thinking() -> None:
    """qwen3.8-flash + none → enable_thinking=False。"""
    payload = _apply_extensions("qwen3.8-flash", reasoning_effort="none")
    assert payload.get("enable_thinking") is False
    assert "thinking_budget" not in payload


def test_qwen_extra_payload_budget_zero_removed() -> None:
    """extra_payload thinking_budget=0 不发出，并关闭思考。"""
    payload = _apply_extensions(
        "qwen3.8-flash",
        extra_payload={"thinking_budget": 0},
    )
    assert "thinking_budget" not in payload
    assert payload.get("enable_thinking") is False


def test_qwen_extra_payload_budget_high_string_removed() -> None:
    """extra_payload thinking_budget='high' 字符串不原样发出。"""
    payload = _apply_extensions(
        "qwen3.8-flash",
        extra_payload={"thinking_budget": "high"},
    )
    assert "thinking_budget" not in payload


def test_qwen_extra_payload_budget_clamped() -> None:
    """extra_payload thinking_budget 超上限钳到 262144。"""
    payload = _apply_extensions(
        "qwen3.8-flash",
        extra_payload={"thinking_budget": 999_999},
    )
    assert payload["thinking_budget"] == QWEN_THINKING_BUDGET_MAX


def test_qwen38_max_high_keeps_reasoning_effort() -> None:
    """qwen3.8-max + high 仍发 reasoning_effort。"""
    payload = _apply_extensions("qwen3.8-max", reasoning_effort="high")
    assert payload.get("reasoning_effort") == "high"
    assert "thinking_budget" not in payload


def test_qwen38_max_budget_wins_over_effort() -> None:
    """qwen3.8-max + effort + extra budget：只保留 budget。"""
    payload = _apply_extensions(
        "qwen3.8-max",
        reasoning_effort="high",
        extra_payload={"thinking_budget": 1024},
    )
    assert payload["thinking_budget"] == 1024
    assert "reasoning_effort" not in payload


def test_non_qwen_high_unchanged() -> None:
    """非 Qwen 模型 high 仍发 reasoning_effort。"""
    payload = _apply_extensions("gpt-4o", reasoning_effort="high")
    assert payload.get("reasoning_effort") == "high"
