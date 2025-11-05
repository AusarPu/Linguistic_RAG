#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ragas 自定义 LLM 封装：OpenAI 兼容 HTTP 客户端（支持 n>1）。

目的：
- 直接调用 vLLM/OpenAI 兼容接口的 /v1/chat/completions，传递 n 参数以支持多样本生成。
- 适配 Ragas BaseRagasLLM 抽象方法：generate_text/agenerate_text/is_finished。

使用：
- 在评估脚本中用 RagasOpenAICompatLLMWrapper 取代 LangchainLLMWrapper(ChatOpenAI)。

注意：
- 遵循用户规则，不使用 try/except 或防错型 if/else。
"""

import json
import urllib.request
import asyncio
import time
import logging
from typing import List, Optional
import aiohttp

from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import Generation, LLMResult
from preprocess.vllm_tokenizer import fast_token_length


class RagasOpenAICompatLLMWrapper(BaseRagasLLM):
    """OpenAI 兼容 HTTP 客户端封装，适配 Ragas BaseRagasLLM 接口。"""

    def __init__(self, base_url: str, model: str, api_key: str = "-", temperature: float = 0.2, top_p: float = 0.9):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logging.getLogger(__name__)
        # 引入 aiohttp 会话与并发信号量（参数从全局配置读取）
        from script.config_rag import EVALUATION_CONCURRENCY_LIMIT
        self._sem = asyncio.Semaphore(EVALUATION_CONCURRENCY_LIMIT)
        # 注意：我们不在此处创建 session，因为 __init__ 可能在非事件循环上下文中
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def timeout(self) -> Optional[int]:
        return getattr(self.run_config, "timeout", None)

    def _http_post_json(self, url: str, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout or 60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _estimate_token_stats(self, messages: List[dict]) -> dict:
        """估算 messages 的 token 数量统计（基于本地 fast tokenizer）。"""
        token_counts = []
        for m in messages:
            content = m.get("content", "")
            token_counts.append(fast_token_length(str(content)))
        total_tokens = sum(token_counts)
        return {"per_message": token_counts, "total": total_tokens}

    def _prompt_to_openai_messages(self, prompt) -> List[dict]:
        """将 PromptValue 转换为 OpenAI messages。"""
        messages: List[dict] = []
        if hasattr(prompt, "to_messages"):
            for m in prompt.to_messages():
                t = getattr(m, "type")
                role = "system" if t == "system" else ("assistant" if t == "ai" else "user")
                messages.append({"role": role, "content": getattr(m, "content")})
        else:
            messages.append({"role": "user", "content": prompt.to_string()})
        return messages

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        messages = self._prompt_to_openai_messages(prompt)
        token_stats = self._estimate_token_stats(messages)
        # 引入评估阶段的 max_tokens 限制
        from script.config_rag import EVALUATION_MAX_TOKENS
        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": EVALUATION_MAX_TOKENS,
        }
        if stop is not None:
            payload["stop"] = stop
        self.logger.info(
            f"[RagasLLM] POST /chat/completions 开始: timeout={self.timeout or 60}s, "
            f"messages={len(messages)}, n={n}, temp={payload['temperature']}, top_p={payload['top_p']}, "
            f"tokens_total≈{token_stats['total']}, tokens_per_msg≈{token_stats['per_message']}"
        )
        start_ts = time.time()
        result = self._http_post_json(url, payload)
        elapsed = time.time() - start_ts
        usage = result.get("usage")
        finish_reasons = [ch.get("finish_reason") for ch in result.get("choices", [])]
        self.logger.info(
            f"[RagasLLM] POST /chat/completions 完成: 耗时={elapsed:.3f}s, "
            f"finish_reasons={finish_reasons}, usage={usage}"
        )
        choices = result.get("choices", [])
        gens = [Generation(text=(ch.get("message", {}).get("content") or "")) for ch in choices] or [Generation(text="")]
        return LLMResult(generations=[gens])

    def close(self):
        """关闭底层 aiohttp ClientSession，释放资源。"""
        if self._session is not None:
            asyncio.run(self._session.close())
            self._session = None

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        # 原生异步实现：aiohttp + 信号量控制并发
        url = f"{self.base_url}/chat/completions"
        messages = self._prompt_to_openai_messages(prompt)
        token_stats = self._estimate_token_stats(messages)
        from script.config_rag import EVALUATION_MAX_TOKENS, VLLM_REQUEST_TIMEOUT
        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "max_tokens": EVALUATION_MAX_TOKENS,
        }
        if stop is not None:
            payload["stop"] = stop

        self.logger.info(
            f"[RagasLLM/async] POST /chat/completions 开始: timeout={self.timeout or 60}s, "
            f"messages={len(messages)}, n={n}, temp={payload['temperature']}, top_p={payload['top_p']}, "
            f"tokens_total≈{token_stats['total']}, tokens_per_msg≈{token_stats['per_message']}"
        )

        # 准备 aiohttp session（懒创建，保持复用）
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout or 60, connect=10.0, sock_read=self.timeout or 60)
            connector = aiohttp.TCPConnector(limit=None)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_ts = time.time()
        await self._sem.acquire()
        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                result_text = await resp.text()
                result = json.loads(result_text)
        finally:
            self._sem.release()

        elapsed = time.time() - start_ts
        usage = result.get("usage")
        finish_reasons = [ch.get("finish_reason") for ch in result.get("choices", [])]
        self.logger.info(
            f"[RagasLLM/async] POST /chat/completions 完成: 耗时={elapsed:.3f}s, "
            f"finish_reasons={finish_reasons}, usage={usage}"
        )
        choices = result.get("choices", [])
        gens = [Generation(text=(ch.get("message", {}).get("content") or "")) for ch in choices] or [Generation(text="")]
        return LLMResult(generations=[gens])


class RagasOpenAICompatEmbeddings(BaseRagasEmbeddings):
    """OpenAI 兼容 Embeddings 客户端（/v1/embeddings），用于 Ragas 评估。

    特性：
    - 原生异步 HTTP（aiohttp），通过 asyncio.Semaphore 控制并发（默认读取配置 EVALUATION_CONCURRENCY_LIMIT）。
    - 不进行分片：embed_documents/aembed_documents 直传完整文本列表（input=list[str]）。
    - 诊断日志：输入文本数量、估算 token 总量与每条分布、请求耗时等。
    - 兼容 Ragas 接口：实现 embed_query/embed_documents 以及异步版本。
    """

    def __init__(self, base_url: str, model: str, api_key: str = "-"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        from script.config_rag import EVALUATION_CONCURRENCY_LIMIT
        self._sem = asyncio.Semaphore(EVALUATION_CONCURRENCY_LIMIT)
        self._session: Optional[aiohttp.ClientSession] = None

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    @property
    def timeout(self) -> Optional[int]:
        return getattr(self, "run_config", None) and getattr(self.run_config, "timeout", None)

    def _http_post_json(self, url: str, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout or 60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _estimate_token_stats_list(self, texts: List[str]) -> dict:
        counts = [fast_token_length(str(t)) for t in texts]
        return {"per_text": counts, "total": sum(counts)}

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": [text]}
        token_stats = self._estimate_token_stats_list([text])
        self.logger.info(
            f"[RagasEmb] POST /embeddings 开始(sync): timeout={self.timeout or 60}s, texts=1, tokens_total≈{token_stats['total']}, tokens_per_text≈{token_stats['per_text']}"
        )
        start_ts = time.time()
        result = self._http_post_json(url, payload)
        elapsed = time.time() - start_ts
        data = result.get("data", [])
        self.logger.info(
            f"[RagasEmb] POST /embeddings 完成(sync): 耗时={elapsed:.3f}s, items={len(data)}"
        )
        emb = data[0].get("embedding", []) if data else []
        return emb

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        token_stats = self._estimate_token_stats_list(texts)
        self.logger.info(
            f"[RagasEmb] POST /embeddings 开始(sync): timeout={self.timeout or 60}s, texts={len(texts)}, tokens_total≈{token_stats['total']}, tokens_per_text≈{token_stats['per_text']}"
        )
        start_ts = time.time()
        result = self._http_post_json(url, payload)
        elapsed = time.time() - start_ts
        data = result.get("data", [])
        self.logger.info(
            f"[RagasEmb] POST /embeddings 完成(sync): 耗时={elapsed:.3f}s, items={len(data)}"
        )
        embs = [item.get("embedding", []) for item in data]
        return embs

    def close(self):
        """关闭底层 aiohttp ClientSession，释放资源。"""
        if self._session is not None:
            asyncio.run(self._session.close())
            self._session = None

    async def aembed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": [text]}
        token_stats = self._estimate_token_stats_list([text])
        self.logger.info(
            f"[RagasEmb/async] POST /embeddings 开始: timeout={self.timeout or 60}s, texts=1, tokens_total≈{token_stats['total']}, tokens_per_text≈{token_stats['per_text']}"
        )

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout or 60, connect=10.0, sock_read=self.timeout or 60)
            connector = aiohttp.TCPConnector(limit=None)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_ts = time.time()
        await self._sem.acquire()
        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                result_text = await resp.text()
                result = json.loads(result_text)
        finally:
            self._sem.release()

        elapsed = time.time() - start_ts
        data = result.get("data", [])
        self.logger.info(
            f"[RagasEmb/async] POST /embeddings 完成: 耗时={elapsed:.3f}s, items={len(data)}"
        )
        emb = data[0].get("embedding", []) if data else []
        return emb

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        token_stats = self._estimate_token_stats_list(texts)
        self.logger.info(
            f"[RagasEmb/async] POST /embeddings 开始: timeout={self.timeout or 60}s, texts={len(texts)}, tokens_total≈{token_stats['total']}, tokens_per_text≈{token_stats['per_text']}"
        )

        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout or 60, connect=10.0, sock_read=self.timeout or 60)
            connector = aiohttp.TCPConnector(limit=None)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start_ts = time.time()
        await self._sem.acquire()
        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                result_text = await resp.text()
                result = json.loads(result_text)
        finally:
            self._sem.release()

        elapsed = time.time() - start_ts
        data = result.get("data", [])
        self.logger.info(
            f"[RagasEmb/async] POST /embeddings 完成: 耗时={elapsed:.3f}s, items={len(data)}"
        )
        embs = [item.get("embedding", []) for item in data]
        return embs