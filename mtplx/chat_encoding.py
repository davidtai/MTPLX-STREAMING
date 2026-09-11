"""Backend-aware chat prompt encoding helpers.

This module is deliberately MLX-free. Public surfaces that need to turn chat
messages into token ids should come through here instead of each surface
recreating Qwen-shaped assumptions.
"""

from __future__ import annotations

import json
from typing import Any

GEMMA4_THINK_OPEN = "<|channel>"
GEMMA4_THINK_CLOSE = "<channel|>"
GEMMA4_THOUGHT_PREFIX = "thought\n"
GEMMA4_EMPTY_THOUGHT_BLOCK = (
    f"{GEMMA4_THINK_OPEN}{GEMMA4_THOUGHT_PREFIX}{GEMMA4_THINK_CLOSE}"
)
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"


def is_gemma4_tokenizer(tokenizer: Any) -> bool:
    try:
        special = getattr(tokenizer, "model_specific_special_tokens", None) or {}
    except Exception:
        special = {}
    try:
        if (
            special.get("think_token") == "<|think|>"
            and special.get("soc_token") == "<|channel>"
            and special.get("eoc_token") == "<channel|>"
        ):
            return True
        if (
            str(getattr(tokenizer, "think_token", "")) == "<|think|>"
            and str(getattr(tokenizer, "soc_token", "")) == "<|channel>"
            and str(getattr(tokenizer, "eoc_token", "")) == "<channel|>"
        ):
            return True
        vocab = tokenizer.get_vocab()
        return all(
            token in vocab
            for token in ("<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>")
        )
    except Exception:
        return False


def encode_without_added_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def strip_gemma4_thinking_text(text: str) -> str:
    while GEMMA4_THINK_OPEN in text and GEMMA4_THINK_CLOSE in text:
        start = text.find(GEMMA4_THINK_OPEN)
        end = text.find(GEMMA4_THINK_CLOSE, start)
        if end < 0:
            break
        text = text[:start] + text[end + len(GEMMA4_THINK_CLOSE) :]
    while QWEN_THINK_OPEN in text and QWEN_THINK_CLOSE in text:
        start = text.find(QWEN_THINK_OPEN)
        end = text.find(QWEN_THINK_CLOSE, start)
        if end < 0:
            break
        text = text[:start] + text[end + len(QWEN_THINK_CLOSE) :]
    return text


def _tool_instruction_message(tools: list[dict[str, Any]]) -> dict[str, str]:
    lines = [
        "Tool calling is available. If a tool is needed, respond with exactly one XML tool call in this format:",
        "<tool_call>",
        "<function=TOOL_NAME>",
        "<parameter=ARGUMENT_NAME>",
        "ARGUMENT_VALUE",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "",
        "Available tools:",
    ]
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or "").strip()
        parameters = function.get("parameters") or {}
        try:
            parameters_text = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        except TypeError:
            parameters_text = str(parameters)
        line = f"- {name}"
        if description:
            line += f": {description}"
        lines.append(line)
        lines.append(f"  parameters: {parameters_text}")
    return {"role": "system", "content": "\n".join(lines)}


def _prepend_tool_instruction(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return messages
    return [_tool_instruction_message(tools), *messages]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    pieces.append(str(item.get("text") or ""))
                elif "text" in item:
                    pieces.append(str(item.get("text") or ""))
            elif item is not None:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content)


def _tool_call_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {"arguments": raw}


def _tool_argument_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _gemma4_tool_call_text(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name") or "").strip()
    if not name:
        return ""
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in _tool_call_arguments(function.get("arguments")).items():
        argument_name = str(key).strip()
        if not argument_name:
            continue
        lines.extend(
            [
                f"<parameter={argument_name}>",
                _tool_argument_text(value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _gemma4_tool_calls_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    rendered: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        text = _gemma4_tool_call_text(tool_call)
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def encode_gemma4_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> list[int]:
    """Encode Gemma 4 text chat turns when a converted artifact has no template."""

    bos = str(getattr(tokenizer, "bos_token", None) or "<bos>")
    parts: list[str] = [bos]
    body_messages = list(messages)
    system_text = ""

    if body_messages and str(body_messages[0].get("role") or "") in {
        "system",
        "developer",
    }:
        system_text = _content_to_text(body_messages[0].get("content")).strip()
        body_messages = body_messages[1:]

    if enable_thinking or system_text:
        parts.append("<|turn>system\n")
        if enable_thinking:
            parts.append("<|think|>\n")
        if system_text:
            parts.append(system_text)
        parts.append("<turn|>\n")

    for item in body_messages:
        role = str(item.get("role") or "")
        content = _content_to_text(item.get("content"))
        if role == "assistant":
            role = "model"
            content = strip_gemma4_thinking_text(content)
            tool_call_text = _gemma4_tool_calls_text(item.get("tool_calls"))
            if tool_call_text:
                content = (
                    f"{content.rstrip()}\n{tool_call_text}"
                    if content.strip()
                    else tool_call_text
                )
        elif role == "tool":
            role = "tool_response"
        if role not in {"user", "model", "tool_response"}:
            continue
        parts.append(f"<|turn>{role}\n")
        if content:
            parts.append(content if role == "user" else content.strip())
        parts.append("<turn|>\n")

    if add_generation_prompt:
        parts.append("<|turn>model\n")
        if not enable_thinking:
            parts.append(GEMMA4_EMPTY_THOUGHT_BLOCK)

    return encode_without_added_special_tokens(tokenizer, "".join(parts))


# ===========================================================================
# DeepSeek-V4.1-Flash chat encoding (W52).
#
# A faithful, MLX-free port of the official reference encoder
# (DeepSeek-V4.1-Flash-src/encoding/encoding.py::encode_messages, text path).
# It is the server-side code fallback for the deepseek_v41 family so a
# checkpoint that ships no chat template still gets the correct render + a
# leading BOS (id 0), never the plain "user:/assistant:" base-model render.
# The installed chat_template.jinja reproduces the identical bytes; this port
# is the parity twin (both validated byte-for-byte against the reference's
# encoding/tests vectors 1-4 and the README quickstart).
#
# Out of scope (text serving path): vision/image content blocks (dropped) and
# the namespace-description merge on tool *schemas* (tool CALLS keep namespaces).
# ===========================================================================

DEEPSEEK_V41_BOS = "<｜begin▁of▁sentence｜>"
DEEPSEEK_V41_EOS = "<｜end▁of▁sentence｜>"
#: DeepSeek-V4.1 begin-of-sentence id. The tokenizer maps the BOS string above
#: to this id; both the render (via the literal token) and the completions path
#: (via a prepend) rely on it. add_bos_token is false, so nothing else adds it.
DEEPSEEK_V41_BOS_ID = 0
_DSV41_THINK_OPEN = "<think>"
_DSV41_THINK_CLOSE = "</think>"
_DSV41_DSML = "｜DSML｜"
_DSV41_USER_SP = "<｜User｜>"
_DSV41_ASSISTANT_SP = "<｜Assistant｜>"
_DSV41_SYSTEM_SP = "<｜System｜>"
_DSV41_REMINDER_SP = "<｜latest_reminder｜>"
_DSV41_TC_BLOCK = " calls"
_DSV41_TC_TAG = " invoke"
_DSV41_TP_TAG = " parameter"
_DSV41_TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
_DSV41_REASONING_EFFORT_MAP = {"low": 50, "high": 75, "max": 100}
_DSV41_DEFAULT_REASONING_EFFORT = "high"
_DSV41_REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
_DSV41_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}{tc_block}>" block like the following:

<{dsml}{tc_block}>
<{dsml}{tc_tag} name="$TOOL_NAME">
<{dsml}{tp_tag} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}{tp_tag}>
...
</{dsml}{tc_tag}>
<{dsml}{tc_tag} name="$TOOL_NAME2">
...
</{dsml}{tc_tag}>
</{dsml}{tc_block}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_open}), you MUST output your complete reasoning inside {think_open}...{think_close} BEFORE any tool calls or final response.

Otherwise, output directly after {think_close} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""
_DSV41_RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
#: The special tokens whose joint presence in the vocab identifies a
#: DeepSeek-V4.1-family tokenizer. All are DeepSeek-only (Qwen uses
#: ``<|im_start|>``, Gemma ``<start_of_turn>``, Step its own set), so the set
#: never collides with the other served families. ``｜DSML｜`` is the DeepSeek
#: tool-markup token.
_DSV41_SIGNATURE_TOKENS = (
    "<｜begin▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜System｜>",
    "｜DSML｜",
)


def is_deepseek_v41_tokenizer(tokenizer: Any) -> bool:
    """True for a DeepSeek-V4.1-family tokenizer (signature special tokens).

    Deliberately tokenizer-only (mirrors :func:`is_gemma4_tokenizer`): the
    encode helpers receive a tokenizer, not the model config. The signature is
    DeepSeek-specific and does not match Qwen/Gemma/Step. (It does not by
    itself separate V4 from V4.1; only V4.1 is served here, and both share the
    render below.)
    """
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return False
    try:
        return all(token in vocab for token in _DSV41_SIGNATURE_TOKENS)
    except Exception:
        return False


def _dsv41_to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def _dsv41_split_tool_name(name: str, namespace: Any = None):
    prefix, sep, bare = str(name).partition("::")
    if sep:
        namespace, name = prefix, bare
    return namespace, name


def _dsv41_tool_name_for_encoding(tool: dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace.get("name")
    namespace, name = _dsv41_split_tool_name(tool.get("name", ""), namespace)
    return name if namespace is None else f"{namespace}::{name}"


def _dsv41_tools_from_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    for tool in tools:
        function = dict(tool.get("function") or {})
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _dsv41_tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = (
                namespace["description"] + "\n" + (function.get("description") or "")
            )
        functions.append(function)
    return functions


def _dsv41_render_tools(tools: list[dict[str, Any]]) -> str:
    schema = "\n".join(_dsv41_to_json(t) for t in _dsv41_tools_from_openai(tools))
    return _DSV41_TOOLS_TEMPLATE.format(
        tool_schemas=schema,
        dsml=_DSV41_DSML,
        tc_block=_DSV41_TC_BLOCK,
        tc_tag=_DSV41_TC_TAG,
        tp_tag=_DSV41_TP_TAG,
        think_open=_DSV41_THINK_OPEN,
        think_close=_DSV41_THINK_CLOSE,
    )


def _dsv41_encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        parsed = arguments
        for _ in range(2):
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except Exception:
                    break
            else:
                break
        arguments = parsed if isinstance(parsed, dict) else {"arguments": tool_call.get("arguments")}
    out: list[str] = []
    for key, value in arguments.items():
        is_str = isinstance(value, str)
        rendered = value if is_str else _dsv41_to_json(value)
        out.append(
            f'<{_DSV41_DSML}{_DSV41_TP_TAG} name="{key}" '
            f'string="{"true" if is_str else "false"}">{rendered}'
            f"</{_DSV41_DSML}{_DSV41_TP_TAG}>"
        )
    return "\n".join(out)


def _dsv41_tool_calls_from_openai(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tc in tool_calls:
        function = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        namespace, name = _dsv41_split_tool_name(
            function.get("name", ""), tc.get("namespace") or function.get("namespace")
        )
        call = {"name": name, "arguments": function.get("arguments")}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def _dsv41_content_text(content: Any) -> str:
    """Text-only flatten (server text path): list content -> joined text, images dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n\n".join(parts)
    return str(content)


def _dsv41_merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge ``tool`` messages (and consecutive users) into user turns.

    Mirrors encoding.py::merge_tool_messages for the text path. Tool results
    become ``<tool_result>...</tool_result>`` inside a user turn; the
    call-order sort is a no-op for a single result and applied by the caller.
    """
    merged: list[dict[str, Any]] = []
    for raw in messages:
        msg = dict(raw)
        role = msg.get("role")
        if role == "tool":
            block = _DSV41_TOOL_RESULT(_dsv41_content_text(msg.get("content", "")))
            if merged and merged[-1].get("role") == "user" and merged[-1].get("_tool_result"):
                merged[-1] = {
                    "role": "user",
                    "content": merged[-1]["content"] + "\n\n" + block,
                    "_tool_result": True,
                }
            else:
                merged.append({"role": "user", "content": block, "_tool_result": True})
        elif role == "user":
            content = _dsv41_content_text(msg.get("content"))
            if (
                merged
                and merged[-1].get("role") == "user"
                and merged[-1].get("task") is None
            ):
                merged[-1] = {
                    **merged[-1],
                    "content": merged[-1]["content"] + "\n\n" + content,
                    "task": msg.get("task"),
                }
            else:
                new_msg = dict(msg)
                new_msg["content"] = content
                new_msg.setdefault("_tool_result", False)
                merged.append(new_msg)
        else:
            merged.append(msg)
    return merged


def _DSV41_TOOL_RESULT(content: str) -> str:
    return f"<tool_result>{content}</tool_result>"


def _dsv41_find_last_user_index(messages: list[dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "user" or (role == "system" and idx > 0):
            return idx
    return -1


def _dsv41_reasoning_effort_prefix(
    index: int, thinking_mode: str, effort: Any
) -> str:
    if effort is None:
        effort = _DSV41_DEFAULT_REASONING_EFFORT
    if isinstance(effort, str) and effort in _DSV41_REASONING_EFFORT_MAP:
        budget: Any = _DSV41_REASONING_EFFORT_MAP[effort]
    else:
        budget = effort
    if index == 0 and thinking_mode == "thinking":
        return _DSV41_REASONING_EFFORT_TEMPLATE.format(budget=budget)
    return ""


def _dsv41_drop_thinking_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_user_idx = _dsv41_find_last_user_index(messages)
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    result: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            trimmed = dict(msg)
            trimmed.pop("reasoning_content", None)
            result.append(trimmed)
    return result


def _dsv41_render_message(
    index: int,
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str,
    drop_thinking: bool,
    reasoning_effort: Any,
    add_generation_prompt: bool,
) -> str:
    msg = messages[index]
    last_user_idx = _dsv41_find_last_user_index(messages)
    role = msg.get("role")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")

    re_prefix = _dsv41_reasoning_effort_prefix(index, thinking_mode, reasoning_effort)
    prompt = _DSV41_SYSTEM_SP if index == 0 and (re_prefix or role == "system") else ""
    prompt += re_prefix

    if role == "system":
        if index > 0:
            prompt += _DSV41_SYSTEM_SP
        prompt += msg.get("content") or ""
        if tools:
            prompt += "\n\n" + _dsv41_render_tools(tools)
        if response_format:
            prompt += "\n\n" + _DSV41_RESPONSE_FORMAT_TEMPLATE.format(
                schema=_dsv41_to_json(response_format)
            )
    elif role == "user":
        prompt += _DSV41_USER_SP + (msg.get("content") or "")
    elif role == "latest_reminder":
        prompt += _DSV41_REMINDER_SP + (msg.get("content") or "")
    elif role == "assistant":
        thinking_part = ""
        tc_content = ""
        if tool_calls:
            calls = _dsv41_tool_calls_from_openai(tool_calls)
            rendered_calls = [
                f'<{_DSV41_DSML}{_DSV41_TC_TAG} name="{_dsv41_tool_name_for_encoding(c)}">\n'
                f"{_dsv41_encode_arguments_to_dsml(c)}\n"
                f"</{_DSV41_DSML}{_DSV41_TC_TAG}>"
                for c in calls
            ]
            tc_content = (
                "\n\n"
                + f"<{_DSV41_DSML}{_DSV41_TC_BLOCK}>\n"
                + "\n".join(rendered_calls)
                + f"\n</{_DSV41_DSML}{_DSV41_TC_BLOCK}>"
            )
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (msg.get("reasoning_content") or "") + _DSV41_THINK_CLOSE
        prompt += thinking_part + (msg.get("content") or "") + tc_content
        if not msg.get("wo_eos", False):
            prompt += DEEPSEEK_V41_EOS
    else:
        raise ValueError(f"deepseek_v41: unknown role {role!r}")

    # Transition: header emitted only when this is the last message or the next
    # message is assistant/latest_reminder (encoding.py::render_message).
    if index + 1 < len(messages) and messages[index + 1].get("role") not in (
        "assistant",
        "latest_reminder",
    ):
        return prompt

    task = msg.get("task")
    is_last = index + 1 >= len(messages)
    if task is not None:
        task_token = _DSV41_TASK_TOKENS[task]
        if task != "action":
            prompt += task_token
        else:
            prompt += _DSV41_ASSISTANT_SP
            prompt += _DSV41_THINK_OPEN if thinking_mode == "thinking" else _DSV41_THINK_CLOSE
            prompt += task_token
    elif role == "user" or (role == "system" and index > 0):
        if is_last and not add_generation_prompt:
            return prompt
        prompt += _DSV41_ASSISTANT_SP
        if (not drop_thinking and thinking_mode == "thinking") or (
            drop_thinking and thinking_mode == "thinking" and index >= last_user_idx
        ):
            prompt += _DSV41_THINK_OPEN
        else:
            prompt += _DSV41_THINK_CLOSE
    return prompt


def render_deepseek_v41_prompt(
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    reasoning_effort: Any = None,
    add_generation_prompt: bool = True,
    drop_thinking: bool = True,
) -> str:
    """Render DeepSeek-V4.1 messages to the reference prompt STRING (BOS-first).

    ``drop_thinking`` defaults True (reference default); it is forced False when
    any message defines tools. Text-only: vision content is flattened to text.
    """
    thinking_mode = "thinking" if enable_thinking else "chat"
    prepared = _dsv41_merge_tool_messages(messages)
    effective_drop = drop_thinking
    if any(m.get("tools") for m in prepared):
        effective_drop = False
    if thinking_mode == "thinking" and effective_drop:
        prepared = _dsv41_drop_thinking_messages(prepared)
    prompt = DEEPSEEK_V41_BOS
    for idx in range(len(prepared)):
        prompt += _dsv41_render_message(
            idx,
            prepared,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt


def encode_deepseek_v41_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None = None,
    add_generation_prompt: bool = True,
    preserve_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Encode DeepSeek-V4.1 chat turns to token ids (BOS id 0 first).

    The code fallback for a deepseek_v41 checkpoint whose tokenizer ships no
    chat template. ``preserve_thinking`` maps to the reference's inverse
    ``drop_thinking`` (preserve → keep history); tools force history kept. A
    top-level ``tools`` list is attached to the first system message when no
    message already carries tools (OpenAI convention), mirroring the template.
    """
    if not messages:
        messages = [{"role": "user", "content": ""}]
    prepared = [dict(m) for m in messages]
    if tools and not any(m.get("tools") for m in prepared):
        attached = False
        for m in prepared:
            if m.get("role") == "system":
                m["tools"] = tools
                attached = True
                break
        if not attached:
            prepared = [{"role": "system", "content": "", "tools": tools}, *prepared]
    rendered = render_deepseek_v41_prompt(
        prepared,
        enable_thinking=bool(enable_thinking),
        reasoning_effort=reasoning_effort,
        add_generation_prompt=add_generation_prompt,
        drop_thinking=not preserve_thinking,
    )
    return encode_without_added_special_tokens(tokenizer, rendered)


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None = None,
    add_generation_prompt: bool = True,
    preserve_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    if not messages:
        messages = [{"role": "user", "content": ""}]
    thinking = bool(enable_thinking)
    if is_gemma4_tokenizer(tokenizer):
        return encode_gemma4_messages(
            tokenizer,
            _prepend_tool_instruction(messages, tools),
            enable_thinking=thinking,
            add_generation_prompt=add_generation_prompt,
        )

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "preserve_thinking": preserve_thinking,
    }
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        template_kwargs["tools"] = tools
    try:
        return list(tokenizer.apply_chat_template(messages, **template_kwargs))
    except TypeError:
        fallback_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            fallback_kwargs["tools"] = tools
        return list(tokenizer.apply_chat_template(messages, **fallback_kwargs))
    except Exception:
        if getattr(tokenizer, "chat_template", None):
            raise
        prompt = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
        )
        if add_generation_prompt:
            prompt += "\nassistant:"
        return list(tokenizer.encode(prompt))
