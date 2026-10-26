#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Router — OpenAI-совместимый прокси-роутер с fallback между провайдерами и моделями.

Возможности:
  - Единая точка входа в стиле OpenAI API: /v1/chat/completions, /v1/models, /v1/embeddings
  - Провайдеры задаются в config.json: base_url + api_key + список моделей
    (Mistral, Ollama, Groq, NVIDIA NIM, Cerebras — любой OpenAI-совместимый)
  - Fallback-роутинг: если модель/провайдер не ответил (таймаут, 429, 5xx, 401/403),
    запрос автоматически уходит на следующую модель в цепочке
  - Поддержка стриминга (SSE), fallback срабатывает до первого отправленного байта
  - Защита эндпоинтов API-ключом: Authorization: Bearer <key> или x-api-key
  - Горячая перезагрузка config.json при изменении файла (без рестарта)

Запуск:
    pip install -r requirements.txt
    python server.py                                  # конфиг берётся из ./config.json
    CONFIG_PATH=/путь/config.json python server.py    # свой путь к конфигу
    PORT=9000 python server.py                        # PORT из окружения имеет приоритет

Переменные окружения (перекрывают config.json — удобно для Render/ClawCloud,
чтобы не держать ключи в репозитории):
    ROUTER_API_KEY    — ключ доступа к самому роутеру (server.api_key)
    REQUIRE_AUTH      — true/false
    PROVIDERS_JSON    — полный JSON-массив провайдеров (заменяет секцию providers)
    На каждого провайдера из конфига (имя -> верхний регистр):
      <ИМЯ>_API_KEY   — ключ провайдера, напр. MISTRAL_API_KEY
      <ИМЯ>_BASE_URL  — базовый URL, напр. MISTRAL_BASE_URL
      <ИМЯ>_MODELS    — модели через запятую, напр. MISTRAL_MODELS=mistral-small-latest,ministral-8b-latest
      <ИМЯ>_ENABLED   — true/false, напр. OLLAMA_ENABLED=false
"""

import asyncio
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("llm-router")
START_TS = time.time()

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).resolve().parent / "config.json"))

CONFIG: dict = {}
_config_mtime: float = -1.0
STATS: dict = {}

# Статусы, при которых роутер переключается на следующую модель.
# Остальные (400, 422 и т.п.) — ошибка самого запроса, отдаются клиенту как есть.
FALLBACK_STATUSES = {401, 403, 404, 408, 410, 429, 500, 502, 503, 504}


# ----------------------------------------------------------------------------
# Конфиг (с горячей перезагрузкой по mtime)
# ----------------------------------------------------------------------------

def load_config(force: bool = False) -> dict:
    global CONFIG, _config_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        log.error("Конфиг не найден: %s", CONFIG_PATH)
        return CONFIG
    if force or not CONFIG or mtime != _config_mtime:
        try:
            CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            _config_mtime = mtime
            log.info("Конфиг загружен: %s", CONFIG_PATH)
        except (json.JSONDecodeError, OSError) as e:
            log.error("Ошибка чтения конфига (%s оставлен старый): %s", "он " if CONFIG else "нет ", e)
    return apply_env_overrides(CONFIG)


def _env_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def apply_env_overrides(cfg: dict) -> dict:
    """Перекрывает конфиг переменными окружения (для деплоя без секретов в репо)."""
    server = cfg.setdefault("server", {})
    if os.environ.get("ROUTER_API_KEY"):
        server["api_key"] = os.environ["ROUTER_API_KEY"].strip()
    if os.environ.get("REQUIRE_AUTH"):
        server["require_auth"] = _env_bool(os.environ["REQUIRE_AUTH"])
    if os.environ.get("REQUEST_TIMEOUT"):
        try:
            server["request_timeout"] = float(os.environ["REQUEST_TIMEOUT"])
        except ValueError:
            pass
    if os.environ.get("PROVIDERS_JSON"):
        try:
            parsed = json.loads(os.environ["PROVIDERS_JSON"])
            if isinstance(parsed, list):
                cfg["providers"] = parsed
                log.info("providers заменены из PROVIDERS_JSON (%d шт.)", len(parsed))
        except json.JSONDecodeError as e:
            log.error("PROVIDERS_JSON не распарсился: %s", e)
    for p in cfg.get("providers", []):
        name = str(p.get("name", "")).upper().replace("-", "_").replace(".", "_")
        if not name:
            continue
        if os.environ.get(f"{name}_API_KEY"):
            p["api_key"] = os.environ[f"{name}_API_KEY"].strip()
        if os.environ.get(f"{name}_BASE_URL"):
            p["base_url"] = os.environ[f"{name}_BASE_URL"].strip()
        if os.environ.get(f"{name}_MODELS"):
            p["models"] = [m.strip() for m in os.environ[f"{name}_MODELS"].split(",") if m.strip()]
        if os.environ.get(f"{name}_ENABLED"):
            p["enabled"] = _env_bool(os.environ[f"{name}_ENABLED"])
    return cfg


def get_providers(cfg: dict) -> list:
    """Включённые провайдеры с base_url. Провайдер без api_key пропускается
    (нечем авторизоваться) — чтобы можно было держать всех включёнными и просто
    добавлять ключи через переменные окружения."""
    out = []
    for p in cfg.get("providers", []):
        if not p.get("enabled", True) or not p.get("base_url"):
            continue
        if not str(p.get("api_key", "") or "").strip():
            log.info("провайдер '%s' пропущен: не задан api_key", p.get("name"))
            continue
        out.append(p)
    return out


def build_chain(cfg: dict) -> list:
    """Полная цепочка (провайдер, модель) в порядке конфига."""
    result = []
    for p in get_providers(cfg):
        for m in p.get("models", []):
            result.append((p, m))
    return result


def build_candidates(cfg: dict, requested_model: Optional[str]) -> list:
    """
    Если клиент указал model — сначала кандидаты с этим именем
    (в форматах 'provider/model' или просто 'model'), затем весь остальной chain.
    """
    full = build_chain(cfg)
    if not requested_model:
        return full
    req = requested_model.strip()
    head, tail, seen = [], [], set()
    for p, m in full:
        key = f"{p.get('name')}::{m}"
        if key in seen:
            continue
        if req in (f"{p.get('name')}/{m}", m):
            head.append((p, m))
            seen.add(key)
    for p, m in full:
        key = f"{p.get('name')}::{m}"
        if key not in seen:
            tail.append((p, m))
            seen.add(key)
    return head + tail


# ----------------------------------------------------------------------------
# Статистика провайдеров (для /v1/providers)
# ----------------------------------------------------------------------------

def note_stat(name: str, ok: bool, latency_ms: Optional[int], model: Optional[str], err: Optional[str] = None) -> None:
    s = STATS.setdefault(name, {"requests": 0, "errors": 0, "last_error": None,
                                "last_model": None, "last_latency_ms": None, "last_used": None})
    if ok:
        s["requests"] += 1
        s["last_latency_ms"] = latency_ms
    else:
        s["errors"] += 1
        s["last_error"] = err
    s["last_model"] = model
    s["last_used"] = int(time.time())


# ----------------------------------------------------------------------------
# Ошибки и авторизация
# ----------------------------------------------------------------------------

def api_error(status: int, message: str, code: str = "api_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "router_error", "code": code}},
    )


def check_auth(request: Request, cfg: dict) -> Optional[JSONResponse]:
    """Bearer <ключ из config.server.api_key> либо заголовок x-api-key. Иначе 401."""
    server = cfg.get("server", {})
    if not server.get("require_auth", True):
        return None
    key = str(server.get("api_key", "") or "")
    if not key:
        return None  # ключ не задан — защита выключена
    token = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    elif request.headers.get("x-api-key"):
        token = request.headers["x-api-key"].strip()
    elif request.query_params.get("key"):
        token = request.query_params["key"].strip()   # для клиентов, не умеющих заголовки
    if token and hmac.compare_digest(token, key):
        return None
    return JSONResponse(
        status_code=401,
        content={"error": {
            "message": "Missing or invalid API key. Use header: Authorization: Bearer <key from config.json>",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }},
    )


def should_fallback(status: Optional[int], exc: Optional[Exception]) -> bool:
    if exc is not None:
        return True
    return status in FALLBACK_STATUSES


def upstream_error_payload(status: int, raw: bytes) -> dict:
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        text = raw.decode("utf-8", errors="replace")[:500]
        return {"error": {"message": text or f"upstream HTTP {status}", "type": "upstream_error", "code": status}}


def router_headers(name: str, model: str, attempts_log: list) -> dict:
    return {
        "x-router-provider": name,
        "x-router-model": model,
        "x-router-attempts": str(len(attempts_log) + 1),
    }


# ----------------------------------------------------------------------------
# Приложение
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config(force=True)
    app.state.http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    yield
    await app.state.http.aclose()


app = FastAPI(title="LLM Router", version="1.4.0", lifespan=lifespan, redoc_url=None)


@app.get("/")
async def root():
    return {
        "service": "llm-router",
        "version": "1.4.0",
        "endpoints": ["/v1", "/v1/models", "/v1/chat/completions", "/v1/embeddings", "/v1/providers",
                       "/api/chat", "/api/generate", "/api/tags", "/api/version", "/api/ps",
                       "/health"],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_sec": round(time.time() - START_TS)}


@app.get("/v1")
async def v1():
    return {"message": "v1"}


@app.get("/v1/models")
async def list_models(request: Request):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    client: httpx.AsyncClient = request.app.state.http
    return {"object": "list", "data": await _fetch_all_models(cfg, client)}


async def _fetch_all_models(cfg: dict, client: httpx.AsyncClient) -> list:
    """Собирает модели со всех провайдеров; недоступные — из конфига."""
    providers = get_providers(cfg)

    async def fetch(p: dict):
        url = p["base_url"].rstrip("/") + "/models"
        headers = {"authorization": f"Bearer {p['api_key']}"} if p.get("api_key") else {}
        try:
            r = await client.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                js = r.json()
                return [(m.get("id"), p.get("name")) for m in js.get("data", []) if m.get("id")]
        except Exception as e:
            log.warning("models: провайдер %s недоступен (%s)", p.get("name"), type(e).__name__)
        # апстрим недоступен — отдаём модели из конфига
        return [(m, p.get("name")) for m in p.get("models", [])]

    results = await asyncio.gather(*(fetch(p) for p in providers))
    data, seen = [], set()
    for pairs in results:
        for mid, owner in pairs:
            if not mid or mid in seen:
                continue
            seen.add(mid)
            data.append({"id": mid, "object": "model", "created": 0, "owned_by": owner})
    return data


@app.get("/v1/providers")
async def providers_status(request: Request):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    out = []
    for p in cfg.get("providers", []):
        out.append({
            "name": p.get("name"),
            "base_url": p.get("base_url"),
            "enabled": p.get("enabled", True),
            "models": p.get("models", []),
            "stats": STATS.get(p.get("name"), {}),
        })
    return {"object": "list", "data": out}


async def relay_post(request: Request, path: str, allow_stream: bool):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return api_error(400, "Invalid JSON in request body")
    if not isinstance(body, dict):
        return api_error(400, "Request body must be a JSON object")

    requested = body.get("model")
    requested_model = requested.strip() if isinstance(requested, str) and requested.strip() else None
    stream = allow_stream and bool(body.get("stream"))
    candidates = build_candidates(cfg, requested_model)
    if not candidates:
        return api_error(503, "No enabled providers/models in config.json")

    server_cfg = cfg.get("server", {})
    default_timeout = float(server_cfg.get("request_timeout", 120))
    connect_timeout = float(server_cfg.get("connect_timeout", 10))
    max_attempts = int(server_cfg.get("max_attempts", 0)) or len(candidates)
    max_attempts = min(max_attempts, len(candidates))

    client: httpx.AsyncClient = request.app.state.http
    attempts_log: list = []

    for provider, model in candidates[:max_attempts]:
        name = provider.get("name", "?")
        url = provider["base_url"].rstrip("/") + path
        headers = {"authorization": f"Bearer {provider['api_key']}"} if provider.get("api_key") else {}
        payload = dict(body)
        payload["model"] = model
        timeout = httpx.Timeout(float(provider.get("timeout") or default_timeout), connect=connect_timeout)
        t0 = time.monotonic()

        if stream:
            req = client.build_request("POST", url, json=payload, headers=headers, timeout=timeout)
            try:
                up = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                attempts_log.append(f"{name}/{model}: {type(e).__name__}")
                note_stat(name, False, None, model, type(e).__name__)
                log.warning("fallback: %s/%s -> %s", name, model, type(e).__name__)
                continue
            if up.status_code != 200:
                raw = await up.aread()
                await up.aclose()
                attempts_log.append(f"{name}/{model}: HTTP {up.status_code}")
                note_stat(name, False, None, model, f"HTTP {up.status_code}")
                log.warning("fallback: %s/%s -> HTTP %s", name, model, up.status_code)
                if should_fallback(up.status_code, None):
                    continue
                return JSONResponse(status_code=up.status_code, content=upstream_error_payload(up.status_code, raw))
            note_stat(name, True, round((time.monotonic() - t0) * 1000), model)

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in up.aiter_raw():
                        yield chunk
                finally:
                    await up.aclose()

            return StreamingResponse(
                relay(),
                media_type=up.headers.get("content-type", "text/event-stream"),
                headers=router_headers(name, model, attempts_log),
            )

        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            attempts_log.append(f"{name}/{model}: {type(e).__name__}")
            note_stat(name, False, None, model, type(e).__name__)
            log.warning("fallback: %s/%s -> %s", name, model, type(e).__name__)
            continue

        latency = round((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                attempts_log.append(f"{name}/{model}: invalid json from upstream")
                note_stat(name, False, latency, model, "bad-json")
                continue
            note_stat(name, True, latency, model)
            return JSONResponse(data, headers=router_headers(name, model, attempts_log))

        attempts_log.append(f"{name}/{model}: HTTP {resp.status_code}")
        note_stat(name, False, latency, model, f"HTTP {resp.status_code}")
        log.warning("fallback: %s/%s -> HTTP %s", name, model, resp.status_code)
        if not should_fallback(resp.status_code, None):
            return JSONResponse(status_code=resp.status_code,
                                content=upstream_error_payload(resp.status_code, resp.content))

    return JSONResponse(status_code=502, content={
        "error": {
            "message": "All providers/models failed",
            "type": "router_error",
            "code": "all_failed",
            "attempts": attempts_log,
        }
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await relay_post(request, "/chat/completions", allow_stream=True)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await relay_post(request, "/embeddings", allow_stream=False)


# ----------------------------------------------------------------------------
# Ollama-совместимые эндпоинты (/api/*)
# Роутер притворяется сервером Ollama: клиенты для Ollama (Open WebUI, LobeChat,
# Continue и т.д.) указывают base_url роутера как Ollama URL.
# Авторизация та же: Bearer / x-api-key / ?key=  (у Ollama-клиентов часто нет ключа —
# тогда отключается REQUIRE_AUTH=false или добавляется ?key= в URL).
# ----------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _ollama_options(body: dict) -> dict:
    opts = body.get("options") or {}
    out = {}
    if opts.get("temperature") is not None:
        out["temperature"] = opts["temperature"]
    if opts.get("top_p") is not None:
        out["top_p"] = opts["top_p"]
    if opts.get("top_k") is not None:
        out["top_k"] = opts["top_k"]
    if opts.get("num_predict") is not None:
        out["max_tokens"] = opts["num_predict"]
    if opts.get("stop"):
        out["stop"] = opts["stop"]
    if body.get("format") == "json":
        out["response_format"] = {"type": "json_object"}
    return out


def _chat_payload_from_ollama(body: dict, model: str, stream: bool) -> dict:
    payload = {"model": model,
               "messages": body.get("messages") or [{"role": "user", "content": ""}],
               "stream": stream}
    payload.update(_ollama_options(body))
    return payload


def _generate_payload_from_ollama(body: dict, model: str, stream: bool) -> dict:
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append({"role": "user", "content": body.get("prompt") or ""})
    payload = {"model": model, "messages": messages, "stream": stream}
    payload.update(_ollama_options(body))
    return payload


def _ollama_chat_from_openai(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    out = {"model": model, "created_at": _iso_now(),
           "message": {"role": msg.get("role", "assistant"), "content": msg.get("content") or ""},
           "done": True}
    if usage.get("prompt_tokens") is not None:
        out["prompt_eval_count"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        out["eval_count"] = usage["completion_tokens"]
    return out


def _ollama_generate_from_openai(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    out = {"model": model, "created_at": _iso_now(),
           "response": (choice.get("message") or {}).get("content") or "", "done": True}
    if usage.get("prompt_tokens") is not None:
        out["prompt_eval_count"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        out["eval_count"] = usage["completion_tokens"]
    return out


async def _openai_sse_to_ollama(up: httpx.Response, model: str, mode: str) -> AsyncIterator[bytes]:
    """Конвертация SSE-потока OpenAI в NDJSON-поток Ollama."""
    try:
        async for raw in up.aiter_lines():
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            content = delta.get("content")
            if not content:
                continue
            if mode == "chat":
                item = {"model": model, "created_at": _iso_now(),
                        "message": {"role": "assistant", "content": content}, "done": False}
            else:
                item = {"model": model, "created_at": _iso_now(), "response": content, "done": False}
            yield (json.dumps(item, ensure_ascii=False) + "\n").encode()
    finally:
        await up.aclose()
    done = {"model": model, "created_at": _iso_now(), "done": True}
    if mode == "generate":
        done["response"] = ""
    yield (json.dumps(done) + "\n").encode()


async def relay_ollama(request: Request, mode: str):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return api_error(400, "Invalid JSON in request body")
    if not isinstance(body, dict):
        return api_error(400, "Request body must be a JSON object")

    requested = body.get("model")
    requested_model = requested.strip() if isinstance(requested, str) and requested.strip() else None
    stream = bool(body.get("stream", True))   # у Ollama stream по умолчанию true
    candidates = build_candidates(cfg, requested_model)
    if not candidates:
        return api_error(503, "No enabled providers/models in config.json")

    server_cfg = cfg.get("server", {})
    default_timeout = float(server_cfg.get("request_timeout", 120))
    connect_timeout = float(server_cfg.get("connect_timeout", 10))
    max_attempts = int(server_cfg.get("max_attempts", 0)) or len(candidates)
    max_attempts = min(max_attempts, len(candidates))

    client: httpx.AsyncClient = request.app.state.http
    attempts_log: list = []
    convert_req = _chat_payload_from_ollama if mode == "chat" else _generate_payload_from_ollama

    for provider, model in candidates[:max_attempts]:
        name = provider.get("name", "?")
        url = provider["base_url"].rstrip("/") + "/chat/completions"
        headers = {"authorization": f"Bearer {provider['api_key']}"} if provider.get("api_key") else {}
        payload = convert_req(body, model, stream)
        timeout = httpx.Timeout(float(provider.get("timeout") or default_timeout), connect=connect_timeout)
        t0 = time.monotonic()

        if stream:
            req = client.build_request("POST", url, json=payload, headers=headers, timeout=timeout)
            try:
                up = await client.send(req, stream=True)
            except httpx.HTTPError as e:
                attempts_log.append(f"{name}/{model}: {type(e).__name__}")
                note_stat(name, False, None, model, type(e).__name__)
                log.warning("fallback: %s/%s -> %s", name, model, type(e).__name__)
                continue
            if up.status_code != 200:
                raw = await up.aread()
                await up.aclose()
                attempts_log.append(f"{name}/{model}: HTTP {up.status_code}")
                note_stat(name, False, None, model, f"HTTP {up.status_code}")
                log.warning("fallback: %s/%s -> HTTP %s", name, model, up.status_code)
                if should_fallback(up.status_code, None):
                    continue
                return JSONResponse(status_code=up.status_code, content=upstream_error_payload(up.status_code, raw))
            note_stat(name, True, round((time.monotonic() - t0) * 1000), model)
            return StreamingResponse(
                _openai_sse_to_ollama(up, model, mode),
                media_type="application/x-ndjson",
                headers=router_headers(name, model, attempts_log),
            )

        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            attempts_log.append(f"{name}/{model}: {type(e).__name__}")
            note_stat(name, False, None, model, type(e).__name__)
            log.warning("fallback: %s/%s -> %s", name, model, type(e).__name__)
            continue
        latency = round((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                attempts_log.append(f"{name}/{model}: invalid json")
                note_stat(name, False, latency, model, "bad-json")
                continue
            note_stat(name, True, latency, model)
            out = _ollama_chat_from_openai(data, model) if mode == "chat" else _ollama_generate_from_openai(data, model)
            return JSONResponse(out, headers=router_headers(name, model, attempts_log))
        attempts_log.append(f"{name}/{model}: HTTP {resp.status_code}")
        note_stat(name, False, latency, model, f"HTTP {resp.status_code}")
        log.warning("fallback: %s/%s -> HTTP %s", name, model, resp.status_code)
        if not should_fallback(resp.status_code, None):
            return JSONResponse(status_code=resp.status_code,
                                content=upstream_error_payload(resp.status_code, resp.content))

    return JSONResponse(status_code=502, content={
        "error": {"message": "All providers/models failed", "type": "router_error",
                  "code": "all_failed", "attempts": attempts_log}
    })


@app.post("/api/chat")
async def api_chat(request: Request):
    return await relay_ollama(request, "chat")


@app.post("/api/generate")
async def api_generate(request: Request):
    return await relay_ollama(request, "generate")


@app.get("/api/version")
async def api_version():
    return {"version": "1.4.0"}


@app.get("/api/ps")
async def api_ps():
    return {"models": []}


@app.get("/api/tags")
async def api_tags(request: Request):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    client: httpx.AsyncClient = request.app.state.http
    data = await _fetch_all_models(cfg, client)
    return {"models": [{"name": m["id"], "model": m["id"], "size": 0, "digest": "",
                         "details": {"family": m.get("owned_by", "")}} for m in data]}


@app.post("/api/embeddings")
async def api_embeddings_legacy(request: Request):
    cfg = load_config()
    denied = check_auth(request, cfg)
    if denied:
        return denied
    try:
        body = await request.json()
    except Exception:
        return api_error(400, "Invalid JSON in request body")
    requested = body.get("model")
    requested_model = requested.strip() if isinstance(requested, str) and requested.strip() else None
    candidates = build_candidates(cfg, requested_model)
    if not candidates:
        return api_error(503, "No enabled providers/models in config.json")
    server_cfg = cfg.get("server", {})
    timeout = httpx.Timeout(float(server_cfg.get("request_timeout", 120)),
                            connect=float(server_cfg.get("connect_timeout", 10)))
    client: httpx.AsyncClient = request.app.state.http
    for provider, model in candidates:
        url = provider["base_url"].rstrip("/") + "/embeddings"
        headers = {"authorization": f"Bearer {provider['api_key']}"} if provider.get("api_key") else {}
        payload = {"model": model, "input": body.get("prompt") or body.get("input") or ""}
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                emb = (resp.json().get("data") or [{}])[0].get("embedding")
                if emb is not None:
                    return JSONResponse({"embedding": emb},
                                        headers=router_headers(provider.get("name", "?"), model, []))
        except httpx.HTTPError:
            continue
    return api_error(502, "All providers failed for embeddings")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = load_config(force=True)
    server = cfg.get("server", {}) if cfg else {}
    host = os.environ.get("HOST", server.get("host", "0.0.0.0"))
    port = int(os.environ.get("PORT", server.get("port", 8000)))
    log.info("LLM Router стартует на http://%s:%s (конфиг: %s)", host, port, CONFIG_PATH)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
