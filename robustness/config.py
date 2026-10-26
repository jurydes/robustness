"""Все настройки стенда - из .env / окружения."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # агент пользователя
    agent_path: str
    agent_func: str
    agent_root: str = ""  # корень импорта проекта; по умолчанию папка файла агента
    agent_env_file: str = ""  # .env проекта, грузится ДО импорта агента
    agent_verdict_field: str = "has_additional_products"
    agent_prompt_attr: str = ""  # атрибут модуля агента с промптом-справочником (необязательно)

    # датасет
    dataset_path: str = "data/dataset.xlsx"
    dataset_text_col: str = "text"
    dataset_label_col: str = "label"
    dataset_pred_col: str = ""  # готовые предсказания на оригиналах; пусто -> агент прогоняется

    out_dir: str = "runs"

    # LLM-генератор (mTLS)
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_ca_bundle_file: str
    gigachat_cert_file: str
    gigachat_key_file: str
    gigachat_model: str = "GigaChat-3-Ultra-preview"
    gigachat_temperature: float = 0.0
    gigachat_timeout: int = 90
    gigachat_max_retries: int = 3
    gigachat_retry_delay: float = 3.0


def path(value: str) -> Path:
    """Относительные пути - от корня стенда."""
    return ROOT / Path(value).expanduser()
