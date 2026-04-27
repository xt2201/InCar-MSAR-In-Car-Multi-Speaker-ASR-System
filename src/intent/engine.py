"""Rule-based Intent Engine for Mandarin Chinese in-car commands."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger


class IntentEngine:
    """Map Mandarin Chinese transcripts to structured intent JSON.

    Uses keyword matching with priority ordering and entity extraction.

    Parameters
    ----------
    keyword_config : str or Path
        Path to intent_keywords.yaml configuration.

    Examples
    --------
    >>> engine = IntentEngine("configs/intent_keywords.yaml")
    >>> engine.parse("降低温度到22度", speaker="Driver")
    {"intent": "climate_decrease", "action": "decrease", "value": 22, "speaker": "Driver"}
    """

    def __init__(self, keyword_config: str | Path = "configs/intent_keywords.yaml") -> None:
        self.keyword_config = Path(keyword_config)
        self._intents: dict[str, dict] = {}
        self._load_keywords()
        logger.info(f"IntentEngine loaded with {len(self._intents)} intent categories.")

    def _load_keywords(self) -> None:
        """Load keyword mapping from YAML config."""
        if not self.keyword_config.exists():
            raise FileNotFoundError(f"Intent config not found: {self.keyword_config}")

        with open(self.keyword_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._intents = config.get("intents", {})

    def parse(
        self,
        transcript: str,
        speaker: str = "unknown",
        confidence_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Parse a transcript into a structured intent.

        Parameters
        ----------
        transcript : str
            Transcribed text in Mandarin Chinese (simplified).
        speaker : str
            Speaker role label (e.g., "Driver", "Passenger_1").
        confidence_threshold : float
            Reserved for future ML-based confidence scoring.

        Returns
        -------
        intent_json : dict with keys:
            "intent" : str – intent category name
            "action" : str – action to perform
            "entity" : str or None – target entity
            "value" : Any or None – extracted value (e.g., temperature number)
            "speaker" : str
            "raw_text" : str
            "matched_keyword" : str or None
        """
        if not transcript or not transcript.strip():
            return self._make_result(
                "unknown", "unknown", None, None, speaker, transcript, None
            )

        # Iterate intent categories in definition order (priority preserved from YAML)
        for intent_name, intent_def in self._intents.items():
            if intent_name == "unknown":
                continue

            keywords = intent_def.get("keywords", [])
            action = intent_def.get("action", "unknown")
            entity = intent_def.get("entity", None)

            for keyword in keywords:
                if re.search(keyword, transcript) if ('.' in keyword or '*' in keyword or '+' in keyword or '\\' in keyword) else (keyword in transcript):
                    # Extract value if pattern defined
                    value = None
                    matched_kw = keyword

                    value_pattern = intent_def.get("value_pattern")
                    if value_pattern:
                        match = re.search(value_pattern, transcript)
                        if match:
                            try:
                                value = int(match.group(1))
                            except (IndexError, ValueError):
                                value = match.group(0)

                    # Extract genre for media intents
                    genre_keywords = intent_def.get("genre_keywords", {})
                    for genre, genre_kws in genre_keywords.items():
                        for gkw in genre_kws:
                            if gkw in transcript:
                                value = genre
                                break

                    # Extract contact for phone intent
                    contact_pattern = intent_def.get("contact_pattern")
                    if contact_pattern:
                        match = re.search(contact_pattern, transcript)
                        if match:
                            groups = [g for g in match.groups() if g]
                            value = groups[0] if groups else None

                    # Extract destination for navigation
                    dest_keywords = intent_def.get("dest_keywords", {})
                    for dest, dest_kws in dest_keywords.items():
                        for dkw in dest_kws:
                            if dkw in transcript:
                                value = dest
                                break

                    return self._make_result(
                        intent_name, action, entity, value, speaker, transcript, matched_kw
                    )

        return self._make_result("unknown", "unknown", None, None, speaker, transcript, None)

    def _make_result(
        self,
        intent: str,
        action: str,
        entity: Optional[str],
        value: Any,
        speaker: str,
        raw_text: str,
        matched_keyword: Optional[str],
    ) -> dict[str, Any]:
        return {
            "intent": intent,
            "action": action,
            "entity": entity,
            "value": value,
            "speaker": speaker,
            "raw_text": raw_text,
            "matched_keyword": matched_keyword,
        }

    def parse_batch(
        self,
        items: list[dict],
    ) -> list[dict[str, Any]]:
        """Parse multiple transcript-speaker pairs.

        Parameters
        ----------
        items : list of dicts with keys "transcript" and "speaker".

        Returns
        -------
        results : list of intent dicts.
        """
        return [
            self.parse(item.get("transcript", ""), item.get("speaker", "unknown"))
            for item in items
        ]

    def supported_intents(self) -> list[str]:
        """Return list of all supported intent categories."""
        return list(self._intents.keys())
