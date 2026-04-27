"""ASR module using OpenAI Whisper via HuggingFace Transformers."""
from __future__ import annotations

import time
from typing import Optional

import torch
import numpy as np
from loguru import logger


class WhisperASR:
    """OpenAI Whisper ASR wrapper for Mandarin Chinese.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "openai/whisper-small".
    language : str
        Target language code, e.g. "zh" for Mandarin.
    device : str
        "cuda" or "cpu".
    beam_size : int
        Beam search width. Smaller = faster, slightly lower quality.
    batch_size : int
        Inference batch size.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        language: str = "zh",
        device: str = "cuda",
        beam_size: int = 2,
        batch_size: int = 1,
        max_new_tokens: int = 448,
    ) -> None:
        self.model_id = model_id
        self.language = language
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._model = None
        self._processor = None

        logger.info(f"WhisperASR initialized: model={model_id}, lang={language}, device={self.device}")

    def _load_model(self) -> None:
        """Lazy-load Whisper via HuggingFace pipeline."""
        if self._pipeline is not None:
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        logger.info(f"Loading Whisper model: {self.model_id}...")

        # float16 only on CUDA; CPU requires float32
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self._model.to(self.device)

        # Transformers ≥5.0 warns when both max_new_tokens and max_length are set.
        # Clear max_length from generation config so max_new_tokens in generate_kwargs
        # is the sole token-limit control (no conflict warning).
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            self._model.generation_config.max_length = None

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        # Transformers ≥5.0: pass language/task at pipeline creation to avoid
        # SuppressTokensLogitsProcessor duplicate warning when language is also in
        # per-call generate_kwargs. The pipeline bakes these into the generation config.
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            chunk_length_s=30,
            batch_size=self.batch_size,
            torch_dtype=dtype,
            device=self.device,
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        logger.info("Whisper loaded successfully.")

    def transcribe(
        self,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
    ) -> dict:
        """Transcribe audio to text.

        Parameters
        ----------
        audio : torch.Tensor or np.ndarray
            Audio waveform, shape [T] or [1, T]. Expected 16kHz.
        sample_rate : int
            Audio sample rate.
        return_timestamps : bool
            If True, return word-level timestamps.

        Returns
        -------
        result : dict with keys:
            "text" : str – transcribed text
            "language" : str
            "inference_time_ms" : float
            "chunks" : list (only if return_timestamps=True)
        """
        self._load_model()

        # Normalize input to numpy [T]
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = np.squeeze(audio)

        if audio_np.ndim != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio_np.shape}")

        # Check for silence (avoid Whisper hallucination on silent chunks)
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < 1e-4:
            logger.debug("Silent chunk detected, returning empty transcript.")
            return {
                "text": "",
                "language": self.language,
                "inference_time_ms": 0.0,
                "is_silent": True,
            }

        t0 = time.perf_counter()

        # language/task are set at pipeline creation; only per-call controls here
        generate_kwargs = {
            "num_beams": self.beam_size,
            "max_new_tokens": min(self.max_new_tokens, 224),
        }

        if return_timestamps:
            generate_kwargs["return_timestamps"] = True

        result = self._pipeline(
            {"array": audio_np, "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        output = {
            "text": result["text"].strip(),
            "language": self.language,
            "inference_time_ms": inference_ms,
            "is_silent": False,
        }

        if return_timestamps and "chunks" in result:
            output["chunks"] = result["chunks"]

        logger.debug(f"ASR: '{output['text'][:50]}...' ({inference_ms:.0f}ms)")
        return output

    def transcribe_batch(
        self,
        audio_list: list[torch.Tensor | np.ndarray],
        sample_rate: int = 16000,
    ) -> list[dict]:
        """Transcribe a batch of audio tensors.

        Parameters
        ----------
        audio_list : list of tensors/arrays
            Each element is a mono audio tensor [T] or [1, T].
        sample_rate : int

        Returns
        -------
        results : list of dicts (same format as transcribe())
        """
        return [self.transcribe(a, sample_rate) for a in audio_list]


class ChunkedASR:
    """Chunked Whisper ASR for streaming inference.

    Processes long audio by splitting into overlapping chunks
    and concatenating transcripts.

    Parameters
    ----------
    asr : WhisperASR
        Underlying ASR instance.
    chunk_sec : float
        Chunk duration in seconds.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        asr: WhisperASR,
        chunk_sec: float = 30.0,
        sample_rate: int = 16000,
    ) -> None:
        self.asr = asr
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_sec * sample_rate)

    def transcribe_long(
        self,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """Transcribe long audio by chunking.

        Parameters
        ----------
        audio : torch.Tensor or np.ndarray
            Shape [T] or [1, T].
        sample_rate : int

        Returns
        -------
        result : dict with keys:
            "text" : str – full concatenated transcript
            "segments" : list of dicts with per-chunk results
            "total_inference_ms" : float
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = np.squeeze(audio)

        total_samples = len(audio_np)
        chunks = [
            audio_np[i : i + self.chunk_samples]
            for i in range(0, total_samples, self.chunk_samples)
        ]

        segments = []
        total_ms = 0.0

        for i, chunk in enumerate(chunks):
            result = self.asr.transcribe(chunk, sample_rate)
            result["chunk_idx"] = i
            result["start_sec"] = i * self.chunk_sec
            result["end_sec"] = min((i + 1) * self.chunk_sec, total_samples / sample_rate)
            segments.append(result)
            total_ms += result["inference_time_ms"]

        full_text = " ".join(
            s["text"] for s in segments if s["text"]
        ).strip()

        return {
            "text": full_text,
            "segments": segments,
            "total_inference_ms": total_ms,
        }
