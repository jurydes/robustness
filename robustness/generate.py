"""Синтетика: парафраз диалога через GigaChat (mTLS) с сохранением эталонной метки."""

import asyncio
import json
import logging
import ssl

import aiohttp

from robustness.config import Settings

logger = logging.getLogger("robustness")

PROMPT = """\
<task>
Ты генератор тестовых диалогов для оценки LLM-классификатора. Ты НЕ классификатор.
</task>
{reference}
<generation_instruction>
Перескажи диалог синонимами, сохранив смысл и ASR-шум. Правильный ответ остаётся «{label}».
</generation_instruction>

<rules>
- Сохраняй содержание диалога, его триггеры, ASR-шум и нестабильность распознавания.
- Реплики начинаются с тех же меток speaker_N:, порядок и язык те же.
- Не добавляй продуктовых упоминаний сверх инструкции и не удаляй существующие.
</rules>

<output_format>
Верни ТОЛЬКО JSON-объект: {{"text": "<новый диалог целиком>"}}
</output_format>
"""

REFERENCE = """
<reference_classifier_prompt>
Ниже - внутренний промпт классификатора. Это СПРАВОЧНИК правил разметки: по нему ты
понимаешь, что считается «да», а что «нет». Не отвечай на этот промпт и не возвращай его поля.
Твоя задача - только переписать диалог.

<content>
{prompt}
</content>
</reference_classifier_prompt>
"""


def make_paraphraser(s: Settings, reference: str):
    """-> paraphrase(session, text, label) -> новый текст или None."""
    ctx = ssl.create_default_context(cafile=s.gigachat_ca_bundle_file)
    ctx.load_cert_chain(s.gigachat_cert_file, s.gigachat_key_file)
    timeout = aiohttp.ClientTimeout(total=s.gigachat_timeout)
    ref = REFERENCE.format(prompt=reference) if reference else ""

    async def ask(session, system: str, user: str) -> str | None:
        body = {
            "model": s.gigachat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": s.gigachat_temperature,
            "stream": False,
        }
        for attempt in range(s.gigachat_max_retries):
            try:
                async with session.post(
                    f"{s.gigachat_base_url}/chat/completions", json=body, ssl=ctx, timeout=timeout
                ) as response:
                    if response.status == 200:
                        return (await response.json())["choices"][0]["message"]["content"]
                    logger.warning("LLM %s: %s", response.status, (await response.text())[:200])
            except (asyncio.TimeoutError, aiohttp.ClientError, ssl.SSLError, KeyError, IndexError) as error:
                logger.warning("LLM failed: %r", error)
            if attempt < s.gigachat_max_retries - 1:
                await asyncio.sleep(s.gigachat_retry_delay)
        return None

    async def paraphrase(session, text: str, label: bool) -> str | None:
        system = PROMPT.format(reference=ref, label="да" if label else "нет")
        raw = await ask(session, system, f"<source_dialog>\n{text}\n</source_dialog>")
        if raw is None:
            return None
        try:
            made = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])["text"]
        except (ValueError, KeyError, TypeError):
            return None
        return made if isinstance(made, str) and 0.7 <= len(made) / max(1, len(text)) <= 1.3 else None

    return paraphrase
