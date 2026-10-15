"""以真实子进程和本地 HTTP/SSE 服务验证 CLI，可复用到打包产物。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STORY = "第一章 来信\n旧信被锁在抽屉里。\n第二章 雨夜\n她终于拆开信封。"
BODY = "门外的脚步停住了。\n\n她握紧信纸，没有出声。"


@pytest.fixture
def cli(tmp_path):
    """每次测试使用独立配置、用户目录和 API 环境变量。"""
    home = tmp_path / "用户目录"
    home.mkdir()
    storage = home / ".novelforge"
    storage.mkdir()
    config = storage / "config.json"
    config.write_text(
        json.dumps({"data": {"storage_path": str(storage)}}), encoding="utf-8"
    )
    story = tmp_path / "前文 空格.txt"
    story.write_text(STORY, encoding="utf-8-sig")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GENGBI_")
    }
    env.update(HOME=str(home), USERPROFILE=str(home), PYTHONIOENCODING="utf-8")
    executable = os.environ.get("GENGBI_TEST_EXECUTABLE")
    command = [executable] if executable else [sys.executable, "-m", "novelforge.cli"]

    def run(*args, stdin=None, extra_env=None):
        return subprocess.run(
            [*command, *map(str, args)],
            cwd=ROOT,
            env={**env, **(extra_env or {})},
            input=stdin,
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=30,
        )

    run.story, run.config, run.storage, run.home = story, config, storage, home
    return run


@pytest.fixture
def api():
    """只替代远端模型，保留真实 socket、客户端解析、预设与文件输出。"""
    state = {"requests": [], "status": 200, "content": BODY, "finish": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body, dict(self.headers)))
            self.send_response(state["status"])
            self.send_header(
                "Content-Type",
                "text/event-stream" if body["stream"] else "application/json",
            )
            self.end_headers()
            if state["status"] != 200:
                self.wfile.write(b'{"error":{"message":"server echoed secret-key"}}')
            elif body["stream"]:
                for char in state["content"]:
                    event = {"choices": [{"delta": {"content": char}}]}
                    data = (
                        "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    ).encode()
                    # 在多字节字符中间写入，覆盖 SSE 增量 UTF-8 解码。
                    split = data.index(char.encode()) + 1
                    self.wfile.write(data[:split])
                    self.wfile.flush()
                    self.wfile.write(data[split:])
                if state["finish"] is not None:
                    event = {
                        "choices": [{"delta": {}, "finish_reason": state["finish"]}]
                    }
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                result = {
                    "choices": [
                        {
                            "message": {"content": state["content"]},
                            "finish_reason": state["finish"],
                        }
                    ]
                }
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_help_and_version(cli):
    assert cli("--help").returncode == 0
    result = cli("--version")
    assert result.returncode == 0 and "0.2.22" in result.stdout
    assert "--lookback" in cli("continue", "--help").stdout


def test_dry_run_uses_default_preset_and_leaves_desktop_data_untouched(cli):
    before = {
        p.relative_to(cli.home): p.read_bytes()
        for p in cli.home.rglob("*")
        if p.is_file()
    }
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--lookback",
        "1",
        "--prompt",
        "不要重复前文",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    messages = "\n".join(item["content"] for item in payload["messages"])
    assert "她终于拆开信封" in messages
    assert "旧信被锁在抽屉里" not in messages
    assert "不要重复前文" in messages
    assert "{{getvar::COT-items}}" not in messages
    assert payload["history_chapter_count"] == 1
    after = {
        p.relative_to(cli.home): p.read_bytes()
        for p in cli.home.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize("stream", ["--stream", "--no-stream"])
def test_real_http_continuation(cli, api, tmp_path, stream):
    api["content"] = (
        "<think>内部推理</think><novelforge_thinking>分析</novelforge_thinking><novelforge_title>回声</novelforge_title>"
        + BODY
    )
    output = tmp_path / "续写 结果.txt"
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--base-url",
        api["url"],
        "--model",
        "test-model",
        stream,
        "--max-tokens",
        "1234",
        "--temperature",
        "0.3",
        "--target-words",
        "1000",
        "--lookback",
        "0",
        "-o",
        output,
        extra_env={"GENGBI_API_KEY": "secret-key"},
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == BODY
    assert result.stdout == ""
    path, request, headers = api["requests"][0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-key"
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 1234 and request["temperature"] == 0.3
    assert request["stream"] is (stream == "--stream")
    assert any(
        "旧信被锁在抽屉里" in message["content"] for message in request["messages"]
    )
    assert "secret-key" not in result.stdout + result.stderr


def test_utf8_stdin_and_stdout(cli, api):
    result = cli(
        "continue",
        "-i",
        "-",
        "--config",
        cli.config,
        "--base-url",
        api["url"],
        "--model",
        "test-model",
        stdin=STORY,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == BODY


def test_legacy_encoding_and_prompt_file(cli, tmp_path):
    cli.story.write_bytes(STORY.encode("gb18030"))
    prompt = tmp_path / "指令.txt"
    prompt.write_text("保持悬念", encoding="utf-8-sig")
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--encoding",
        "gb18030",
        "--prompt-file",
        prompt,
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "保持悬念" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--temperature", "nan"],
        ["--temperature", "3"],
        ["--lookback", "-1"],
        ["--target-words", "0"],
        ["--max-tokens", "0"],
        ["--timeout", "inf"],
        ["--max-context", "10", "--max-tokens", "10"],
        ["--preset", "../escape"],
    ],
)
def test_invalid_options_do_not_request_model(cli, api, arguments):
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--base-url",
        api["url"],
        "--model",
        "test",
        *arguments,
    )
    assert result.returncode == 2, result.stderr
    assert not api["requests"]
    assert "Traceback" not in result.stderr


def test_missing_config_and_empty_input(cli, tmp_path):
    assert cli("presets", "--config", tmp_path / "missing.json").returncode == 2
    result = cli("continue", "-i", "-", "--dry-run", stdin="  ")
    assert result.returncode == 2


def test_existing_output_and_input_are_protected(cli, api, tmp_path):
    output = tmp_path / "existing.txt"
    output.write_text("保留内容", encoding="utf-8")
    args = ["continue", "-i", cli.story, "--base-url", api["url"], "--model", "test"]
    assert cli(*args, "-o", output).returncode == 2
    assert cli(*args, "-o", cli.story, "--force").returncode == 2
    assert not api["requests"]
    assert output.read_text(encoding="utf-8") == "保留内容"
    assert cli(*args, "-o", output, "--force").returncode == 0
    assert output.read_text(encoding="utf-8").strip() == BODY


@pytest.mark.parametrize(
    "status,content,finish",
    [
        (401, BODY, "stop"),
        (500, BODY, "stop"),
        (200, "", "stop"),
        (200, BODY, "length"),
        (200, BODY, None),
    ],
)
def test_failures_preserve_output_and_hide_server_body(
    cli, api, tmp_path, status, content, finish
):
    api.update(status=status, content=content, finish=finish)
    output = tmp_path / "result.txt"
    output.write_text("原内容", encoding="utf-8")
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--base-url",
        api["url"],
        "--model",
        "test",
        "-o",
        output,
        "--force",
    )
    assert result.returncode == 1, result.stderr
    assert "secret-key" not in result.stderr and "Traceback" not in result.stderr
    assert result.stdout == ""
    assert output.read_text(encoding="utf-8") == "原内容"


def test_endpoint_listing_and_selection(cli, api):
    cli.config.write_text(
        json.dumps(
            {
                "data": {"storage_path": str(cli.storage)},
                "default_endpoint_id": "one",
                "api_endpoints": [
                    {
                        "id": "one",
                        "name": "本地服务",
                        "base_url": api["url"],
                        "default_model": "configured",
                        "extra_headers": {"X-Test": "private-header"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = cli("endpoints", "--config", cli.config)
    assert result.returncode == 0 and "private-header" not in result.stdout
    assert json.loads(result.stdout)[0]["id"] == "one"
    result = cli(
        "continue", "-i", cli.story, "--config", cli.config, "--endpoint", "one"
    )
    assert result.returncode == 0, result.stderr
    assert api["requests"][0][1]["model"] == "configured"
    assert api["requests"][0][2]["X-Test"] == "private-header"
    # 切换地址后不能携带旧端点的私有 HTTP 头。
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--base-url",
        api["url"],
        "--model",
        "other",
    )
    assert result.returncode == 0, result.stderr
    assert "X-Test" not in api["requests"][-1][2]


def test_st_preset_file_and_environment_overrides(cli, api, tmp_path):
    preset = tmp_path / "ST 预设.json"
    preset.write_text(
        json.dumps(
            {
                "name": "测试预设",
                "temperature": 0.6,
                "openai_max_tokens": 777,
                "prompts": [
                    {
                        "identifier": "main",
                        "content": "续写 {{book}}，约 {{target_words}} 字。",
                        "role": "system",
                    },
                    {"identifier": "chatHistory", "marker": True},
                ],
                "prompt_order": [
                    {
                        "character_id": 100000,
                        "order": [
                            {"identifier": "main", "enabled": True},
                            {"identifier": "chatHistory", "enabled": True},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--preset-file",
        preset,
        "--title",
        "雨夜",
        "--target-words",
        "800",
        extra_env={"GENGBI_BASE_URL": api["url"], "GENGBI_MODEL": "from-env"},
    )
    assert result.returncode == 0, result.stderr
    request = api["requests"][0][1]
    assert request["model"] == "from-env" and request["max_tokens"] == 777
    assert any(
        "续写 雨夜，约 800 字" in item["content"] for item in request["messages"]
    )
    assert not (cli.storage / "presets").exists()


def test_headless_path_does_not_import_qt(cli):
    if os.environ.get("GENGBI_TEST_EXECUTABLE"):
        pytest.skip("打包产物由构建排除 Qt；此测试专门验证源码导入")
    code = """
import importlib.abc, runpy, sys
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(('PySide', 'PyQt', 'novelforge.ui')):
            raise AssertionError('CLI imported GUI: ' + fullname)
sys.meta_path.insert(0, NoQt())
sys.argv = ['gengbi', 'continue', '-i', sys.argv[1], '--config', sys.argv[2], '--dry-run']
runpy.run_module('novelforge.cli', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(cli.story), str(cli.config)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["messages"]


def test_saved_encrypted_key_is_used_without_rewriting_config(cli, api, monkeypatch):
    from novelforge.core.config import ConfigManager

    monkeypatch.setenv("HOME", str(cli.home))
    monkeypatch.setenv("USERPROFILE", str(cli.home))
    manager = ConfigManager(cli.config)
    manager.load()
    manager.add_endpoint(
        {
            "id": "saved",
            "name": "已保存",
            "base_url": api["url"],
            "api_key": "encrypted-test-key",
            "default_model": "saved-model",
        }
    )
    before = cli.config.read_bytes()
    result = cli("continue", "-i", cli.story, "--config", cli.config)
    assert result.returncode == 0, result.stderr
    assert api["requests"][0][2]["Authorization"] == "Bearer encrypted-test-key"
    assert cli.config.read_bytes() == before
    assert "encrypted-test-key" not in result.stdout + result.stderr


def test_saved_preset_and_regex_toggle(cli, api):
    presets = cli.storage / "presets"
    presets.mkdir()
    from novelforge.utils.paths import get_default_preset_path

    data = json.loads(get_default_preset_path().read_text(encoding="utf-8"))
    data.update(id="custom", name="自定义预设")
    (presets / "custom.json").write_text(json.dumps(data), encoding="utf-8")
    assert {
        row["id"] for row in json.loads(cli("presets", "--config", cli.config).stdout)
    } == {"default", "custom"}
    api["content"] = "<novelforge_thinking>保留测试</novelforge_thinking>" + BODY
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--preset",
        "custom",
        "--base-url",
        api["url"],
        "--model",
        "test",
        "--no-regex",
    )
    assert result.returncode == 0, result.stderr
    assert "保留测试" in result.stdout


def test_keyboard_interrupt_returns_130(monkeypatch, capsys):
    from novelforge import cli as module

    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(module.asyncio, "run", interrupt)
    assert module.main(["presets"]) == 130
    assert "已取消" in capsys.readouterr().err


@pytest.mark.parametrize("data", [{"api_endpoints": "broken"}, {"data": []}, []])
def test_invalid_config_has_controlled_error(cli, data):
    cli.config.write_text(json.dumps(data), encoding="utf-8")
    result = cli("endpoints", "--config", cli.config)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


@pytest.fixture
def jailbreak_preset(tmp_path):
    """用中性标记验证模块选择，故意让提示开关与排序开关冲突。"""
    identifiers = [
        f"nf_jb_{level}" for level in ("low", "mid", "high", "cot", "prefill")
    ]
    data = {
        "id": "test-jailbreak",
        "name": "模块选择测试",
        "generation_params": {},
        "prompts": [
            {
                "identifier": identifier,
                "content": f"CHECK_{identifier}",
                "enabled": False,
            }
            for identifier in identifiers
        ]
        + [{"identifier": "main", "content": "KEEP_MAIN"}],
        "prompt_order": [
            {
                "character_id": 100000,
                "order": [
                    {"identifier": identifier, "enabled": True}
                    for identifier in [*identifiers, "main"]
                ],
            }
        ],
    }
    path = tmp_path / "模块 预设.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("level", [None, "off", "low", "mid", "high", "cot"])
def test_jailbreak_override_reaches_http_without_changing_preset(
    cli, api, jailbreak_preset, level
):
    before = jailbreak_preset.read_bytes(), cli.config.read_bytes()
    override = [] if level is None else ["--jailbreak", level]
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--preset-file",
        jailbreak_preset,
        "--base-url",
        api["url"],
        "--model",
        "test",
        *override,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == BODY
    messages = json.dumps(api["requests"][0][1]["messages"])
    for candidate in ("low", "mid", "high", "cot"):
        assert (f"CHECK_nf_jb_{candidate}" in messages) == (
            level is None or level == candidate
        )
    assert ("CHECK_nf_jb_prefill" in messages) == (level != "off")
    assert "KEEP_MAIN" in messages
    assert (jailbreak_preset.read_bytes(), cli.config.read_bytes()) == before


@pytest.mark.parametrize("source", ["saved", "st"])
def test_jailbreak_supports_saved_and_st_presets(cli, jailbreak_preset, source):
    data = json.loads(jailbreak_preset.read_text(encoding="utf-8"))
    if source == "saved":
        directory = cli.storage / "presets"
        directory.mkdir()
        path = directory / "test-jailbreak.json"
        selection = ["--preset", "test-jailbreak"]
    else:
        data.pop("generation_params")
        path = jailbreak_preset
        selection = ["--preset-file", path]
    path.write_text(json.dumps(data), encoding="utf-8")
    before = {
        p.relative_to(cli.home): p.read_bytes()
        for p in cli.home.rglob("*")
        if p.is_file()
    }
    original = path.read_bytes()
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        *selection,
        "--jailbreak",
        "mid",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    messages = json.dumps(json.loads(result.stdout)["messages"])
    assert "CHECK_nf_jb_mid" in messages and "CHECK_nf_jb_high" not in messages
    assert path.read_bytes() == original
    assert {
        p.relative_to(cli.home): p.read_bytes()
        for p in cli.home.rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize("level", ["off", "low", "mid", "high", "cot"])
def test_jailbreak_builtin_preset(cli, level):
    data = json.loads(
        (ROOT / "novelforge/resources/defaults/default_preset.json").read_text(
            encoding="utf-8"
        )
    )
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--jailbreak",
        level,
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    messages = "\n".join(
        item["content"] for item in json.loads(result.stdout)["messages"]
    )
    for prompt in data["prompts"]:
        if prompt["identifier"] in {
            "nf_jb_low",
            "nf_jb_mid",
            "nf_jb_high",
            "nf_jb_cot",
        }:
            assert (prompt["content"] in messages) == (
                prompt["identifier"] == f"nf_jb_{level}"
            )


@pytest.mark.parametrize("missing", ["prompt", "entry", "group"])
def test_jailbreak_missing_module_fails_before_request(
    cli, api, jailbreak_preset, missing
):
    data = json.loads(jailbreak_preset.read_text(encoding="utf-8"))
    if missing == "prompt":
        data["prompts"] = [
            p for p in data["prompts"] if p["identifier"] != "nf_jb_high"
        ]
    elif missing == "entry":
        data["prompt_order"][0]["order"] = [
            p
            for p in data["prompt_order"][0]["order"]
            if p["identifier"] != "nf_jb_high"
        ]
    else:
        data["prompt_order"][0]["character_id"] = 42
    jailbreak_preset.write_text(json.dumps(data), encoding="utf-8")
    result = cli(
        "continue",
        "-i",
        cli.story,
        "--config",
        cli.config,
        "--preset-file",
        jailbreak_preset,
        "--jailbreak",
        "high",
        "--base-url",
        api["url"],
        "--model",
        "test",
    )
    assert result.returncode == 2
    assert "nf_jb_high" in result.stderr and "Traceback" not in result.stderr
    assert not api["requests"]


def test_jailbreak_invalid_level(cli):
    result = cli("continue", "-i", cli.story, "--jailbreak", "invalid", "--dry-run")
    assert result.returncode == 2
    assert "--jailbreak" in result.stderr and "Traceback" not in result.stderr
