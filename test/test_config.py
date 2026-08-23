from app.config import get_settings


def test_settings_load_without_error():
    settings = get_settings()
    assert settings.kb_dir == "knowledge-base"


def test_paths_resolve_to_absolute():
    settings = get_settings()
    assert settings.kb_dir_path.is_absolute()
    assert settings.kb_dir_path.name == "knowledge-base"


def test_kb_dir_path_exists_in_repo():
    settings = get_settings()
    assert settings.kb_dir_path.is_dir()