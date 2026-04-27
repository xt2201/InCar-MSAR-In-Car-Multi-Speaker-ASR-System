"""Unit tests for IntentEngine with 50 labeled Mandarin test cases."""
import pytest
from src.intent.engine import IntentEngine


@pytest.fixture(scope="module")
def engine():
    return IntentEngine("configs/intent_keywords.yaml")


# 50 test cases: (input_text, expected_intent, speaker)
TEST_CASES = [
    # Climate control
    ("增加温度", "climate_increase", "Driver"),
    ("调高温度一点", "climate_increase", "Driver"),
    ("暖一点", "climate_increase", "Passenger_1"),
    ("开暖气", "climate_increase", "Driver"),
    ("降低温度", "climate_decrease", "Driver"),
    ("调低温度", "climate_decrease", "Passenger_1"),
    ("凉一点", "climate_decrease", "Driver"),
    ("开空调", "climate_decrease", "Driver"),
    ("降温到22度", "climate_decrease", "Driver"),
    ("温度低一点", "climate_decrease", "Passenger_1"),
    # Media play
    ("播放音乐", "media_play", "Driver"),
    ("放音乐", "media_play", "Passenger_1"),
    ("开音乐", "media_play", "Driver"),
    ("播放爵士乐", "media_play", "Driver"),
    ("放摇滚", "media_play", "Passenger_1"),
    ("放歌", "media_play", "Driver"),
    # Media pause
    ("暂停", "media_pause", "Driver"),
    ("停止音乐", "media_pause", "Driver"),
    ("关音乐", "media_pause", "Passenger_1"),
    ("停播", "media_pause", "Driver"),
    # Volume
    ("调大音量", "media_volume_up", "Driver"),
    ("声音大一点", "media_volume_up", "Passenger_1"),
    ("大声点", "media_volume_up", "Driver"),
    ("调小音量", "media_volume_down", "Driver"),
    ("声音小一点", "media_volume_down", "Passenger_1"),
    ("小声点", "media_volume_down", "Driver"),
    # Navigation
    ("导航去公司", "navigation_go", "Driver"),
    ("带我去家", "navigation_go", "Driver"),
    ("回家", "navigation_go", "Driver"),
    ("开始导航", "navigation_go", "Driver"),
    ("取消导航", "navigation_cancel", "Driver"),
    ("关闭导航", "navigation_cancel", "Driver"),
    ("停止导航", "navigation_cancel", "Driver"),
    # Window
    ("打开车窗", "window_open", "Driver"),
    ("开窗", "window_open", "Passenger_1"),
    ("关窗", "window_close", "Driver"),
    ("关车窗", "window_close", "Passenger_1"),
    ("关上窗户", "window_close", "Driver"),
    # Phone
    ("打电话", "phone_call", "Driver"),
    ("给妈妈打电话", "phone_call", "Driver"),
    ("呼叫爸爸", "phone_call", "Driver"),
    # Unknown
    ("", "unknown", "Driver"),
    ("今天天气怎么样", "unknown", "Driver"),
    ("随便说一句话", "unknown", "Passenger_1"),
    ("这是测试句子", "unknown", "Driver"),
    # Additional climate
    ("热一点", "climate_increase", "Passenger_1"),
    ("加热", "climate_increase", "Driver"),
    ("冷一点", "climate_decrease", "Passenger_1"),
    ("减小音量", "media_volume_down", "Driver"),
    ("导航到单位", "navigation_go", "Driver"),
]


class TestIntentEngine:
    def test_engine_loads(self, engine):
        assert engine is not None
        assert len(engine.supported_intents()) > 0

    def test_returns_dict(self, engine):
        result = engine.parse("播放音乐")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "action" in result
        assert "speaker" in result

    def test_climate_increase(self, engine):
        result = engine.parse("增加温度", speaker="Driver")
        assert result["intent"] == "climate_increase"

    def test_media_play(self, engine):
        result = engine.parse("播放音乐", speaker="Driver")
        assert result["intent"] == "media_play"

    def test_temperature_value_extraction(self, engine):
        result = engine.parse("降温到22度", speaker="Driver")
        assert result["intent"] == "climate_decrease"
        assert result["value"] == 22

    def test_unknown_intent(self, engine):
        result = engine.parse("今天天气怎么样", speaker="Driver")
        assert result["intent"] == "unknown"

    def test_empty_transcript(self, engine):
        result = engine.parse("")
        assert result["intent"] == "unknown"

    def test_batch_parse(self, engine):
        items = [
            {"transcript": "播放音乐", "speaker": "Driver"},
            {"transcript": "降低温度", "speaker": "Passenger_1"},
        ]
        results = engine.parse_batch(items)
        assert len(results) == 2
        assert results[0]["intent"] == "media_play"
        assert results[1]["intent"] == "climate_decrease"

    @pytest.mark.parametrize("text,expected_intent,speaker", TEST_CASES)
    def test_labeled_cases(self, engine, text, expected_intent, speaker):
        result = engine.parse(text, speaker=speaker)
        assert result["intent"] == expected_intent, (
            f"Input: '{text}' | Expected: {expected_intent} | Got: {result['intent']}"
        )
