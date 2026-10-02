"""Configuration boundaries derived from the user's global rules 3, 5 and 7."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ecsa.settings import Settings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def test_settings_load_without_credentials_from_another_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_yaml(CONFIG_PATH)
    assert settings.seed == 42
    assert settings.data_dir == CONFIG_PATH.parent.parent / "data"
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.interim_dir == settings.data_dir / "interim"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.cache_dir == settings.data_dir / "cache"
    assert settings.sec_user_agent is None
    assert settings.openai_api_key is None
    assert settings.transcript_api_key is None
    assert settings.storage_timezone == "UTC"
    assert settings.market_timezone == "America/New_York"
    assert settings.llm_enabled is False


@pytest.mark.parametrize(
    ("environment", "property_name"),
    [
        ("SEC_USER_AGENT", "sec_user_agent"),
        ("OPENAI_API_KEY", "openai_api_key"),
        ("TRANSCRIPT_API_KEY", "transcript_api_key"),
    ],
)
def test_secrets_only_from_environment_and_excluded_from_output(
    monkeypatch: pytest.MonkeyPatch, environment: str, property_name: str
) -> None:
    sentinel = "Synthetic Credential only-for-this-test@example.invalid"
    monkeypatch.setenv(environment, sentinel)
    settings = Settings.from_yaml(CONFIG_PATH)
    assert getattr(settings, property_name).get_secret_value() == sentinel
    assert sentinel not in repr(settings)
    assert sentinel not in settings.model_dump_json()
    assert property_name not in settings.model_dump()


@pytest.mark.parametrize("field", ["sec_user_agent", "openai_api_key", "transcript_api_key"])
def test_yaml_rejects_credentials(field: str, tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    payload[field] = "synthetic-secret"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings.from_yaml(path)


def test_dotenv_is_not_an_implicit_secret_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=do-not-load-this\n")
    monkeypatch.chdir(tmp_path)
    assert Settings.from_yaml(CONFIG_PATH).openai_api_key is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finbert_revision", "main"),
        ("seed", -1),
        ("finbert_max_tokens", 513),
        ("sec_requests_per_second", 10),
        ("sec_requests_per_second", 0),
        ("request_timeout_seconds", float("inf")),
        ("cache_ttl_seconds", 0),
        ("storage_timezone", "America/New_York"),
        ("market_timezone", "UTC"),
        ("typo_setting", 42),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: object, tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    payload[field] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        Settings.from_yaml(path)


@pytest.mark.parametrize("content", ["", "- data\n- raw\n", "42"])
def test_yaml_requires_mapping(content: str, tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="must contain a mapping"):
        Settings.from_yaml(path)
