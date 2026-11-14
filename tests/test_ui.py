from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_ui_opens_and_scans_without_llm_configuration(sample_repo):
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "ui.py"), default_timeout=30).run()
    assert not app.exception
    app.sidebar.text_input[0].set_value(str(sample_repo))
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any(tab.label == "修改影响" for tab in app.tabs)
    assert any(tab.label == "Evaluation" for tab in app.tabs)
