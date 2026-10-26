"""Агент пользователя: файл + функция из env.

Контракт: `async def f(session, payload) -> dict` или `def f(payload) -> dict`;
payload = {call_id, text, duration, create_dt}; в ответе - булево поле AGENT_VERDICT_FIELD
и (необязательно) status="success".
"""

import asyncio
import importlib
import inspect
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from robustness.config import Settings, path

logger = logging.getLogger("robustness")


def load_agent(s: Settings):
    """-> (predict(session, text) -> bool | None, промпт-справочник или "")."""
    file = path(s.agent_path).resolve()
    root = path(s.agent_root).resolve() if s.agent_root else file.parent
    if s.agent_env_file:  # агент читает настройки при импорте -> env раньше импорта
        load_dotenv(path(s.agent_env_file), override=True)
    sys.path.insert(0, str(root))

    module = importlib.import_module(".".join(file.relative_to(root).with_suffix("").parts))
    fn = getattr(module, s.agent_func)
    required = [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty]
    with_session = len(required) >= 2

    async def predict(session, text: str) -> bool | None:
        payload = {
            "call_id": str(uuid4()),
            "text": text,
            "duration": 0.0,
            "create_dt": datetime.now(timezone.utc).isoformat(),
        }
        args = (session, payload) if with_session else (payload,)
        try:
            result = await fn(*args) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, *args)
        except Exception as error:  # агент - чёрный ящик, сбой = None
            logger.warning("agent failed: %r", error)
            return None
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        if not isinstance(result, dict) or result.get("status", "success") != "success":
            return None
        verdict = result.get(s.agent_verdict_field)
        return verdict if isinstance(verdict, bool) else None

    return predict, str(getattr(module, s.agent_prompt_attr)) if s.agent_prompt_attr else ""
