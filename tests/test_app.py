"""Smoke test: every dashboard page renders, and a simulated day can be clicked through.

Needs the cleaned data and the trained model, so it is skipped until
``python scripts/train_all.py --skip-vision`` has been run.
"""

import pytest

from loomiq.config import PROJECT_ROOT
from loomiq.data.clean import get_clean_data_path
from loomiq.models.train_productivity import get_model_path

pytestmark = pytest.mark.skipif(
    not (get_clean_data_path().exists() and get_model_path().exists()),
    reason="needs cleaned data and a trained model",
)

PAGES = [
    "pages/1_Line_Monitor.py",
    "pages/2_Recommendations.py",
    "pages/3_Defect_Scanner.py",
    "pages/4_AI_Assistant.py",
]


def click(app, label: str) -> None:
    next(button for button in app.sidebar.button if label in button.label).click().run()


def test_all_pages_render_through_a_day() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=120).run()
    assert not app.exception

    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, page

    app.switch_page("pages/1_Line_Monitor.py").run()
    for _ in range(3):
        click(app, "Next hour")
    assert app.session_state.simulator.current_hour == 3
    assert not app.exception

    app.switch_page("pages/2_Recommendations.py").run()
    apply = [button for button in app.button if "Apply" in button.label]
    if apply:
        apply[0].click().run()
        assert not app.exception

    click(app, "End shift")
    assert app.session_state.simulator.is_finished
    for page in ["Home.py", *PAGES]:
        app.switch_page(page).run()
        assert not app.exception, page
