import pytest
from app.models.decision_engine import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine(
        daily_capital_cost_rate=0.0003,
        cost_per_call=15.0,
        avg_days_accelerated=3.0,
        daily_capacity=20,
    )


def test_positive_ev_for_large_late_invoice(engine):
    ev, components = engine.calculate_expected_value(
        invoice_amount=300000,
        predicted_days_late=5.0,
        prediction_lower=0.0,
        prediction_upper=10.0,
        customer_responsiveness=0.6,
    )
    assert ev > 0, f"Expected positive EV for large late invoice, got {ev}"
    assert components["p_late"] > 0.5


def test_negative_ev_for_small_early_invoice(engine):
    ev, components = engine.calculate_expected_value(
        invoice_amount=100,
        predicted_days_late=-10.0,
        prediction_lower=-15.0,
        prediction_upper=-5.0,
        customer_responsiveness=0.5,
    )
    assert ev < 0, f"Expected negative EV for small early invoice, got {ev}"
    assert components["p_late"] == 0.05


def test_p_late_when_upper_negative(engine):
    _, components = engine.calculate_expected_value(
        invoice_amount=1000,
        predicted_days_late=-10,
        prediction_lower=-20,
        prediction_upper=-3,
        customer_responsiveness=0.5,
    )
    assert components["p_late"] == 0.05


def test_p_late_when_lower_above_5(engine):
    _, components = engine.calculate_expected_value(
        invoice_amount=1000,
        predicted_days_late=20,
        prediction_lower=10,
        prediction_upper=30,
        customer_responsiveness=0.5,
    )
    assert components["p_late"] == 0.95


def test_p_late_interpolation(engine):
    _, components = engine.calculate_expected_value(
        invoice_amount=1000,
        predicted_days_late=2,
        prediction_lower=-5,
        prediction_upper=5,
        customer_responsiveness=0.5,
    )
    assert 0 < components["p_late"] < 1
    assert components["p_late"] == pytest.approx(0.5, abs=0.01)


def test_call_cost_deducted(engine):
    ev, components = engine.calculate_expected_value(
        invoice_amount=0,
        predicted_days_late=10,
        prediction_lower=5,
        prediction_upper=15,
        customer_responsiveness=0.5,
    )
    assert ev == -15.0  # Benefit is 0, minus call cost
