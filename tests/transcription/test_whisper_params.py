from transcription.parameters import get_default_whisper_params


def test_default_whisper_params_beam_size_comes_from_env(monkeypatch):
    """WHISPER_BEAM_SIZE 決定 process 預設 beam_size。"""
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "1")
    assert get_default_whisper_params().beam_size == 1
