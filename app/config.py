"""Settings. Environment-driven so the deployment box and a laptop can differ without code
changes — notably `device`, which is "cuda" on the Spark and "cpu" locally."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BAG_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    db_path: Path = REPO_ROOT / "data" / "jobs.sqlite3"

    # --- Hardware ---
    device: str = "cuda"
    compute_type: str = "float16"
    # Two components in this stack (CTranslate2, Chatterbox) fail by silently falling back
    # to CPU rather than erroring. On a batch workload that means shipping something that
    # appears to work and runs ~20x slower. Assert instead.
    require_cuda: bool = True

    # --- ASR ---
    whisper_model: str = "large-v3"   # not -turbo: its dropped decoder layers hurt ru/zh/ja
    vad_filter: bool = True
    align_model: str | None = None    # None lets WhisperX pick per detected language

    # --- Sentence boundary detection ---
    sat_model: str = "sat-3l-sm"

    # --- Translation ---
    llm_base_url: str = "http://localhost:11434/v1"   # Ollama; vLLM is drop-in compatible
    llm_model: str = "qwen3:14b"
    llm_api_key: str = "unused"
    translate_batch_size: int = 10
    translate_context_sentences: int = 2
    llm_timeout: float = 120.0

    # --- TTS ---
    tts_engine: str = "kokoro"
    tts_sample_rate: int = 24000

    # --- Limits ---
    max_video_duration: float = 90 * 60
    job_retention_days: int = 7

    # --- yt-dlp ---
    ytdlp_cookies_from_browser: str | None = None
    ytdlp_format: str = "bestaudio/best"

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id


settings = Settings()
