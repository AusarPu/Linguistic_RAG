#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM 结构化输出测试脚本：验证是否支持在参数中设定返回 JSON 格式。

涵盖三类用法：
1) JSON 模式：response_format={"type": "json_object"}
2) JSON Schema：response_format={"type": "json_schema", "json_schema": {...}}
3) Guided JSON（vLLM 扩展）：extra_body={"guided_json": {...}, "guided_decoding_backend": "auto|outlines|xgrammar|guidance"}

说明：
- 本脚本使用 OpenAI 兼容接口 /v1/chat/completions。
- 为了遵从用户约束，不使用 try/except；若返回内容不是合法 JSON，将直接抛错，便于定位问题。
- 日志初始化采用项目统一策略，只输出 WARNING 及重试相关日志。

用法示例：
  python Reports/experiments/evaluation/test_vllm_json_mode.py \
    --prompt "返回一个包含 name(字符串)、age(整数) 的 JSON" \
    --max-tokens 256 --temperature 0.0 --top-p 0.95 \
    --backend auto

可选参数：
  --prompt: 测试提示词（默认会要求返回 name/age）
  --backend: guided_decoding 后端（auto/outlines/xgrammar/guidance），默认 auto
  --max-tokens / --temperature / --top-p: 采样参数
"""

import argparse
import json
import logging
from typing import Dict

from pydantic import BaseModel
from openai import OpenAI

from script import config_rag as config


class Person(BaseModel):
    name: str
    age: int


def _get_client_and_model() -> OpenAI:
    """基于项目配置初始化 OpenAI 兼容客户端。"""
    base_url = config.GENERATOR_API_URL.rsplit("/chat/completions", 1)[0]
    client = OpenAI(base_url=base_url, api_key="dummy")
    return client


def _build_messages(prompt: str):
    return [
        {"role": "system", "content": "你是一个只输出 JSON 的助手。"},
        {"role": "user", "content": prompt},
    ]


def test_json_object(client: OpenAI, model: str, prompt: str, params: Dict):
    logging.warning("[TEST] 开始 JSON 模式测试：response_format={type=json_object}")
    completion = client.chat.completions.create(
        model=model,
        messages=_build_messages(prompt),
        response_format={"type": "json_object"},
        max_tokens=params.get("max_tokens", 256),
        temperature=params.get("temperature", 0.0),
        top_p=params.get("top_p", 0.95),
    )
    content = completion.choices[0].message.content
    logging.warning(f"[TEST/json_object] 原始返回: {content}")
    parsed = json.loads(content)
    logging.warning(f"[TEST/json_object] 解析成功: {parsed}")


def test_json_schema(client: OpenAI, model: str, prompt: str, params: Dict):
    logging.warning("[TEST] 开始 JSON Schema 测试：response_format={type=json_schema}")
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": Person.model_json_schema(),
        },
    }
    completion = client.chat.completions.create(
        model=model,
        messages=_build_messages(prompt),
        response_format=rf,
        max_tokens=params.get("max_tokens", 256),
        temperature=params.get("temperature", 0.0),
        top_p=params.get("top_p", 0.95),
    )
    content = completion.choices[0].message.content
    logging.warning(f"[TEST/json_schema] 原始返回: {content}")
    model_obj = Person.model_validate_json(content)
    logging.warning(f"[TEST/json_schema] Pydantic 校验成功: {model_obj}")


def test_guided_json(client: OpenAI, model: str, prompt: str, params: Dict, backend: str):
    logging.warning("[TEST] 开始 Guided JSON 测试：extra_body={guided_json,...}")
    extra_body = {
        "guided_json": {
            "name": "person",
            "schema": Person.model_json_schema(),
        },
        "guided_decoding_backend": backend,
    }
    completion = client.chat.completions.create(
        model=model,
        messages=_build_messages(prompt),
        max_tokens=params.get("max_tokens", 256),
        temperature=params.get("temperature", 0.0),
        top_p=params.get("top_p", 0.95),
        extra_body=extra_body,
    )
    content = completion.choices[0].message.content
    logging.warning(f"[TEST/guided_json] 原始返回: {content}")
    parsed = json.loads(content)
    model_obj = Person.model_validate_json(content)
    logging.warning(f"[TEST/guided_json] 解析 + 校验成功: {parsed} / {model_obj}")


def main():
    config.setup_logging()

    parser = argparse.ArgumentParser(description="测试 vLLM JSON 输出支持")
    parser.add_argument(
        "--prompt",
        type=str,
        default="返回一个包含 name(字符串)、age(整数) 的 JSON。name 使用中文名，age 为 18-60 的整数。",
    )
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    client = _get_client_and_model()
    model = config.GENERATOR_MODEL_NAME_FOR_API
    params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    logging.warning("[TEST] 使用的模型: %s", model)
    test_json_object(client, model, args.prompt, params)
    test_json_schema(client, model, args.prompt, params)
    test_guided_json(client, model, args.prompt, params, args.backend)


if __name__ == "__main__":
    main()