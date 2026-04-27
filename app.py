"""Streamlit Demo App – In-Car Multi-Speaker ASR System.

Launch:
    streamlit run app.py
    streamlit run app.py --server.port 8502  # Custom port
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="In-Car Multi-Speaker ASR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
INTENT_ICONS = {
    "climate_increase":  "🌡️↑",
    "climate_decrease":  "🌡️↓",
    "media_play":        "▶️",
    "media_pause":       "⏸️",
    "media_volume_up":   "🔊",
    "media_volume_down": "🔉",
    "navigation_go":     "🗺️",
    "navigation_cancel": "❌",
    "window_open":       "🪟↑",
    "window_close":      "🪟↓",
    "phone_call":        "📞",
    "unknown":           "❓",
}

ROLE_COLORS = {
    "Driver":      "#1f77b4",  # Blue
    "Passenger_1": "#ff7f0e",  # Orange
    "Passenger_2": "#2ca02c",  # Green
    "Passenger_3": "#d62728",  # Red
}


# ─────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────
if "transcript_rows" not in st.session_state:
    st.session_state.transcript_rows = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ─────────────────────────────────────────────────────────────────────
# Helper: Load pipeline (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models (one-time setup)...")
def load_pipeline(config_path: str = "configs/demo.yaml", _n_speakers: int = 2):
    """Load and cache the pipeline (heavy models loaded once)."""
    from src.pipeline.orchestrator import InCarASRPipeline
    pipeline = InCarASRPipeline(config_path)
    pipeline.cfg.separation.n_speakers = _n_speakers
    return pipeline


@st.cache_data(show_spinner=False)
def get_available_samples(data_dir: str) -> list[str]:
    """Return list of available sample files."""
    wav_dir = Path(data_dir) / "wav"
    if not wav_dir.exists():
        return []
    return sorted([f.name for f in wav_dir.glob("*.wav")])[:50]


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 InCar ASR Demo")
    st.markdown("**In-Car Multi-Speaker ASR**")
    st.markdown("Dataset: AISHELL-5 (OpenSLR #159)")
    st.divider()

    config_path = st.selectbox(
        "Configuration",
        ["configs/demo.yaml", "configs/default.yaml"],
        help="demo.yaml = faster (whisper-small, rule-based)"
    )

    data_split = st.selectbox("Dataset Split", ["dev", "eval1", "eval2"])
    data_dir = f"data/{data_split}"

    sample_files = get_available_samples(data_dir)
    if sample_files:
        selected_file = st.selectbox(
            "Select Audio Sample",
            sample_files,
            help="Choose a recording from AISHELL-5"
        )
        wav_path = str(Path(data_dir) / "wav" / selected_file)
    else:
        st.warning(f"No audio files found in `{data_dir}/wav/`")
        st.info("Run: `bash scripts/download_data.sh`")
        wav_path = None
        selected_file = None

    st.divider()

    n_speakers = st.slider("Expected Speakers", min_value=2, max_value=4, value=2)
    show_waveform = st.checkbox("Show Waveform", value=True)
    show_raw_json = st.checkbox("Show Raw JSON Output", value=False)

    st.divider()
    st.markdown("**System Info**")
    cuda_available = torch.cuda.is_available()
    st.markdown(f"GPU: {'✅ Available' if cuda_available else '⚠️ CPU only'}")
    if cuda_available:
        st.markdown(f"Device: `{torch.cuda.get_device_name(0)}`")


# ─────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────
st.title("🚗 In-Car Multi-Speaker ASR")
st.markdown(
    "_System demo – Speech Separation + ASR + Speaker Role + Intent Understanding_"
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎤 Input Audio")

    if wav_path and Path(wav_path).exists():
        # Audio player
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

        # Waveform visualization
        if show_waveform:
            try:
                import torchaudio
                import plotly.graph_objects as go

                waveform, sr = torchaudio.load(wav_path)
                # Show only channel 0 (driver mic)
                ch0 = waveform[0].numpy()
                t = np.linspace(0, len(ch0) / sr, len(ch0))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t[::100], y=ch0[::100],
                    mode="lines",
                    name="Driver mic (ch0)",
                    line=dict(color="#1f77b4", width=1),
                ))
                fig.update_layout(
                    title="Waveform – Driver Microphone (Channel 0)",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Waveform unavailable: {e}")

        # File info
        try:
            import torchaudio
            waveform, sr = torchaudio.load(wav_path)
            dur = waveform.shape[-1] / sr
            st.caption(
                f"Shape: {list(waveform.shape)} | "
                f"SR: {sr}Hz | "
                f"Duration: {dur:.1f}s | "
                f"Channels: {waveform.shape[0]}"
            )
        except Exception:
            pass
    else:
        st.info("No audio file selected. Please download data first.")

    st.divider()

    # Control buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        start_btn = st.button(
            "▶ Start Pipeline",
            type="primary",
            disabled=(wav_path is None or not Path(wav_path).exists() if wav_path else True),
            use_container_width=True,
        )
    with btn_col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    with btn_col3:
        export_btn = st.button("💾 Export JSON", use_container_width=True)


with col_right:
    st.subheader("📝 Transcript & Intent")
    transcript_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────
# Status bar (full width)
# ─────────────────────────────────────────────────────────────────────
status_bar = st.empty()

# ─────────────────────────────────────────────────────────────────────
# Button logic
# ─────────────────────────────────────────────────────────────────────
if reset_btn:
    st.session_state.transcript_rows = []
    st.session_state.processing = False
    status_bar.info("Pipeline reset.")
    st.rerun()

if export_btn and st.session_state.transcript_rows:
    json_str = json.dumps(st.session_state.transcript_rows, ensure_ascii=False, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="transcript_output.json",
        mime="application/json",
    )

if start_btn and wav_path and Path(wav_path).exists():
    st.session_state.processing = True
    st.session_state.transcript_rows = []

    # Load pipeline
    status_bar.info("Loading AI models...")
    try:
        pipeline = load_pipeline(config_path, _n_speakers=n_speakers)
    except Exception as e:
        status_bar.error(f"Failed to load pipeline: {e}")
        st.session_state.processing = False
        st.stop()

    # Process file
    status_bar.info(f"Processing `{selected_file}`...")
    t_start = time.time()

    try:
        utterances = pipeline.process_file(wav_path)
        elapsed = time.time() - t_start
        st.session_state.transcript_rows = utterances
        status_bar.success(
            f"Done in {elapsed:.1f}s – "
            f"{len(utterances)} speakers detected. "
            f"({selected_file})"
        )
    except FileNotFoundError as e:
        status_bar.error(f"File not found: {e}")
    except ValueError as e:
        status_bar.error(f"Audio error: {e}")
    except Exception as e:
        status_bar.error(f"Pipeline error: {e}")
        st.exception(e)
    finally:
        st.session_state.processing = False

# ─────────────────────────────────────────────────────────────────────
# Display transcript table
# ─────────────────────────────────────────────────────────────────────
with col_right:
    if st.session_state.transcript_rows:
        rows = st.session_state.transcript_rows

        # Build styled HTML table
        table_html = f"""
        <style>
        .asr-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .asr-table th {{ background: #2d2d2d; color: white; padding: 8px 12px; text-align: left; }}
        .asr-table td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
        .asr-table tr:hover td {{ background: #f5f5f5; }}
        .badge-driver {{ background: {ROLE_COLORS["Driver"]}; color: white; border-radius: 4px; padding: 2px 8px; }}
        .badge-passenger1 {{ background: {ROLE_COLORS["Passenger_1"]}; color: white; border-radius: 4px; padding: 2px 8px; }}
        .badge-passenger2 {{ background: {ROLE_COLORS["Passenger_2"]}; color: white; border-radius: 4px; padding: 2px 8px; }}
        .badge-passenger3 {{ background: {ROLE_COLORS["Passenger_3"]}; color: white; border-radius: 4px; padding: 2px 8px; }}
        </style>
        <table class="asr-table">
        <tr>
            <th>Speaker</th>
            <th>Transcript</th>
            <th>Intent</th>
        </tr>
        """

        for utt in rows:
            role = utt.get("role", "Unknown")
            transcript = utt.get("transcript", "")
            intent_data = utt.get("intent", {})
            intent = intent_data.get("intent", "unknown") if isinstance(intent_data, dict) else "unknown"
            icon = INTENT_ICONS.get(intent, "❓")

            if "Driver" in role:
                badge_class = "badge-driver"
            elif "_3" in role:
                badge_class = "badge-passenger3"
            elif "_2" in role:
                badge_class = "badge-passenger2"
            else:
                badge_class = "badge-passenger1"

            # Dim silent utterances
            if utt.get("is_silent"):
                transcript = "<em style='color:#999'>(silence)</em>"

            table_html += f"""
            <tr>
                <td><span class="{badge_class}">{role}</span></td>
                <td>{transcript or '<em style="color:#999">—</em>'}</td>
                <td>{icon} {intent}</td>
            </tr>
            """

        table_html += "</table>"
        transcript_placeholder.markdown(table_html, unsafe_allow_html=True)

        if show_raw_json:
            st.subheader("Raw JSON Output")
            st.json(st.session_state.transcript_rows)

    else:
        transcript_placeholder.info(
            "Press **▶ Start Pipeline** to analyze the selected audio sample."
        )

# ─────────────────────────────────────────────────────────────────────
# Processing indicator
# ─────────────────────────────────────────────────────────────────────
if st.session_state.processing:
    with st.spinner("Running pipeline..."):
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "In-Car Multi-Speaker ASR | AISHELL-5 (CC BY-SA 4.0) | "
    "Whisper (MIT) | SpeechBrain (Apache 2.0) | Asteroid (MIT)"
)
