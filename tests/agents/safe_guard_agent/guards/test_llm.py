from prediction_market_agent.agents.safe_watch_agent.watchers.llm import (
    trim_long_string_values,
)


def test_trim_long_values_dict() -> None:
    input_data = {"short": "value", "long": "a" * 1001}
    expected_output = {"short": "value", "long": "N/A (max value length exceeded)"}
    assert trim_long_string_values(input_data, 1000) == expected_output


def test_trim_long_values_list() -> None:
    input_data = ["short", "a" * 1001]
    expected_output = ["short", "N/A (max value length exceeded)"]
    assert trim_long_string_values(input_data, 1000) == expected_output


def test_trim_long_values_string() -> None:
    input_data = "a" * 1001
    expected_output = "N/A (max value length exceeded)"
    assert trim_long_string_values(input_data, 1000) == expected_output


def test_trim_long_values_nested() -> None:
    input_data = {
        "list": ["short", "a" * 1001],
        "dict": {"short": "value", "long": "a" * 1001},
    }
    expected_output = {
        "list": ["short", "N/A (max value length exceeded)"],
        "dict": {"short": "value", "long": "N/A (max value length exceeded)"},
    }
    assert trim_long_string_values(input_data, 1000) == expected_output


def test_trim_long_values_no_trim() -> None:
    input_data = {"short": "value", "another_short": "another_value"}
    expected_output = {"short": "value", "another_short": "another_value"}
    assert trim_long_string_values(input_data, 1000) == expected_output
