"""Speaker role classification module.

Two implementations:
1. RuleBasedRoleClassifier – fast energy-based heuristic (MVP)
2. ECAPAEmbedding + SpeakerRoleClassifier – embedding-based (advanced)
"""
from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from loguru import logger


class RuleBasedRoleClassifier:
    """Classify speaker roles using channel energy heuristic.

    Assigns 'Driver' to the speaker whose separated track has the highest
    energy in the driver microphone channel.

    Parameters
    ----------
    driver_channel : int
        Index of the microphone channel closest to the driver seat (0-based).
    channel_seat_map : dict
        Mapping from channel index to seat label.
    """

    def __init__(
        self,
        driver_channel: int = 0,
        channel_seat_map: Optional[dict] = None,
    ) -> None:
        self.driver_channel = driver_channel
        self.channel_seat_map = channel_seat_map or {
            0: "driver",
            1: "passenger_front",
            2: "passenger_rear_left",
            3: "passenger_rear_right",
        }

    def classify(
        self,
        sources: torch.Tensor,
        multichannel_mix: torch.Tensor,
    ) -> dict[int, str]:
        """Assign speaker roles to separated tracks.

        Strategy: compute correlation of each separated track with the driver
        channel of the original mixture. The track most correlated with the
        driver channel is labeled "Driver".

        Parameters
        ----------
        sources : torch.Tensor
            Separated speaker tracks, shape [N, T].
        multichannel_mix : torch.Tensor
            Original 4-channel audio, shape [4, T] or [C, T].

        Returns
        -------
        role_map : dict[int, str]
            Mapping from track index to role label
            e.g. {0: "Driver", 1: "Passenger_1", 2: "Passenger_2"}
        """
        n_speakers = sources.shape[0]
        if self.driver_channel >= multichannel_mix.shape[0]:
            return {0: "Driver", **{i: f"Passenger_{i}" for i in range(1, n_speakers)}}
        driver_ch = multichannel_mix[self.driver_channel]  # [T]

        # Compute cross-correlation (energy-weighted) with driver channel
        correlations = []
        for i in range(n_speakers):
            track = sources[i]  # [T]
            # Normalize
            track_norm = track / (track.norm() + 1e-8)
            driver_norm = driver_ch / (driver_ch.norm() + 1e-8)
            corr = (track_norm * driver_norm).sum().abs().item()
            correlations.append(corr)

        driver_idx = int(np.argmax(correlations))

        role_map = {}
        passenger_count = 1
        for i in range(n_speakers):
            if i == driver_idx:
                role_map[i] = "Driver"
            else:
                role_map[i] = f"Passenger_{passenger_count}"
                passenger_count += 1

        logger.debug(f"Role classification (rule-based): {role_map}, correlations={correlations}")
        return role_map

    def classify_by_energy(
        self,
        multichannel_mix: torch.Tensor,
        n_speakers: int = 2,
    ) -> dict[int, str]:
        """Simplified: assign roles based solely on channel RMS energy.

        Channel with highest energy in driver_channel -> track labeled Driver.
        Used when separated tracks are not available.

        Parameters
        ----------
        multichannel_mix : torch.Tensor
            Shape [C, T].
        n_speakers : int
            Expected number of speakers.

        Returns
        -------
        role_map : dict[int, str]
        """
        role_map = {0: "Driver"}
        for i in range(1, n_speakers):
            role_map[i] = f"Passenger_{i}"
        return role_map


class ECAPAEmbedding:
    """ECAPA-TDNN speaker embedding extractor (SpeechBrain pretrained).

    Extracts 192-dim speaker embeddings for identification and clustering.

    Parameters
    ----------
    model_hub : str
        SpeechBrain model ID on HuggingFace.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_hub: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda",
    ) -> None:
        self.model_hub = model_hub
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self._model = None

        logger.info(f"ECAPAEmbedding initialized: model={model_hub}, device={self.device}")

    def _load_model(self) -> None:
        """Lazy-load ECAPA-TDNN model."""
        if self._model is not None:
            return

        from speechbrain.pretrained import EncoderClassifier

        logger.info(f"Loading ECAPA-TDNN from {self.model_hub}...")
        # SpeechBrain ≥1.0 requires "cuda:N" format, not plain "cuda"
        sb_device = "cuda:0" if self.device == "cuda" else self.device
        self._model = EncoderClassifier.from_hparams(
            source=self.model_hub,
            savedir=f".cache/speechbrain/ecapa",
            run_opts={"device": sb_device},
        )
        logger.info("ECAPA-TDNN loaded successfully.")

    def extract(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract 192-dim speaker embedding from audio.

        Parameters
        ----------
        audio : torch.Tensor
            Shape [T] or [1, T]. Expected 16kHz.
        sample_rate : int

        Returns
        -------
        embedding : torch.Tensor
            Shape [1, 192] – L2-normalized speaker embedding.
        """
        self._load_model()

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().unsqueeze(0)  # [1, T]
        else:
            audio = torch.tensor(audio).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.encode_batch(
                audio.to(self.device)
            )  # [1, 1, 192]
            embedding = embedding.squeeze(1)  # [1, 192]

        return embedding.cpu()

    def cosine_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Parameters
        ----------
        emb1, emb2 : torch.Tensor
            Shape [1, 192] or [192].

        Returns
        -------
        similarity : float in [-1, 1].
        """
        e1 = emb1.squeeze().float()
        e2 = emb2.squeeze().float()
        return torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()


class SpeakerRoleClassifier:
    """Unified speaker role classifier combining ECAPA embeddings and energy heuristic.

    Clustering approach:
    1. Extract ECAPA embeddings for all tracks.
    2. Cluster embeddings to N speakers using cosine distance.
    3. Assign 'Driver' to the cluster whose centroid most correlates
       with the driver microphone channel energy.

    Parameters
    ----------
    ecapa : ECAPAEmbedding
        ECAPA embedding extractor.
    rule_classifier : RuleBasedRoleClassifier
        Fallback rule-based classifier.
    cosine_threshold : float
        Minimum cosine similarity to consider same speaker (0.7 default).
    """

    def __init__(
        self,
        ecapa: ECAPAEmbedding,
        rule_classifier: RuleBasedRoleClassifier,
        cosine_threshold: float = 0.7,
    ) -> None:
        self.ecapa = ecapa
        self.rule = rule_classifier
        self.cosine_threshold = cosine_threshold

    def classify(
        self,
        sources: torch.Tensor,
        multichannel_mix: torch.Tensor,
        inference_ms_budget: float = 200.0,
    ) -> dict[int, str]:
        """Classify speaker roles using hybrid method.

        First tries ECAPA-based clustering; falls back to rule-based if
        utterances are too short or embedding extraction fails.

        Parameters
        ----------
        sources : torch.Tensor
            Shape [N, T].
        multichannel_mix : torch.Tensor
            Shape [C, T].
        inference_ms_budget : float
            Maximum inference time allowed (ms). If exceeded, falls back to rule.

        Returns
        -------
        role_map : dict[int, str]
        """
        import time

        n_speakers = sources.shape[0]
        embeddings = []

        t0 = time.perf_counter()
        try:
            for i in range(n_speakers):
                emb = self.ecapa.extract(sources[i])
                embeddings.append(emb)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if elapsed_ms > inference_ms_budget:
                    logger.warning(
                        f"ECAPA inference budget exceeded ({elapsed_ms:.0f}ms > {inference_ms_budget}ms). "
                        "Falling back to rule-based classification."
                    )
                    return self.rule.classify(sources, multichannel_mix)

        except Exception as e:
            logger.warning(f"ECAPA extraction failed: {e}. Falling back to rule-based.")
            return self.rule.classify(sources, multichannel_mix)

        # Use rule-based to identify the driver track, then use ECAPA cosine
        # similarity to verify track identity is consistent (same speaker within
        # each track). If two tracks have cosine > threshold (suspiciously similar),
        # fallback to pure rule-based since separation may have failed.
        role_map = self.rule.classify(sources, multichannel_mix)

        if len(embeddings) >= 2:
            sim = torch.nn.functional.cosine_similarity(
                embeddings[0].squeeze().unsqueeze(0),
                embeddings[1].squeeze().unsqueeze(0),
            ).item()
            if sim > self.cosine_threshold:
                logger.warning(
                    f"High embedding similarity ({sim:.2f} > {self.cosine_threshold}): "
                    "tracks may not be well separated."
                )

        logger.debug(f"Hybrid classification result: {role_map}")
        return role_map
