from app.settings import Settings

_REQUIRED = {
    "azure_openai_endpoint": "https://test.openai.azure.com/",
    "azure_openai_api_key": "test-key",
    "azure_openai_api_version": "2024-02-01",
    "azure_openai_deployment": "gpt-4",
    "azure_openai_embeddings_deployment": "text-embedding",
}


def test_guardrail_max_message_length_defaults_to_2000():
    settings = Settings(_env_file=None, **_REQUIRED)
    assert settings.guardrail_max_message_length == 2000


def test_guardrail_enabled_defaults_to_true():
    settings = Settings(_env_file=None, **_REQUIRED)
    assert settings.guardrail_enabled is True


def test_guardrail_max_message_length_is_configurable():
    settings = Settings(_env_file=None, **_REQUIRED, guardrail_max_message_length=500)
    assert settings.guardrail_max_message_length == 500


def test_guardrail_can_be_disabled():
    settings = Settings(_env_file=None, **_REQUIRED, guardrail_enabled=False)
    assert settings.guardrail_enabled is False
