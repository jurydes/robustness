# robustness

    pip install -e .
    cp .env.example .env     # заполнить
    python -m robustness --limit 20

    data/                    # датасет (DATASET_PATH)
    robustness/
      config.py              # Settings из env
      agent.py               # импорт агента по AGENT_PATH + AGENT_FUNC
      generate.py            # парафраз через GigaChat (mTLS)
      metrics.py             # без изменений
      run.py                 # датасет, прогон, CLI

Контракт агента: `async def f(session, payload) -> dict` или `def f(payload) -> dict`;
payload = {call_id, text, duration, create_dt}; в ответе булево поле AGENT_VERDICT_FIELD
и (опц.) status="success".
