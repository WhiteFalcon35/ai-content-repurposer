import os
import tempfile
import subprocess
import streamlit as st
import yt_dlp
from typing import List, Dict

from faster_whisper import WhisperModel
from llm_utils import generate_text
from prompts import (
    twitter_prompt,
    linkedin_prompt,
    reel_prompt,
    refined_transcript_prompt,
    key_takeaways_prompt,
    mistakes_prompt,
    application_prompt,
)

# ============================================================
# SESSION STATE
# ============================================================
SESSION_KEYS = [
    "segments",
    "raw_transcript",
    "refined_transcript",
    "frames",
    "takeaways",
    "mistakes",
    "application",
    "twitter_content",
    "linkedin_content",
    "reel_content",
]

for k in SESSION_KEYS:
    st.session_state.setdefault(k, None)


# ============================================================
# WHISPER
# ============================================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# ============================================================
# YOUTUBE AUDIO DOWNLOAD (AUDIO ONLY – SAFE)
# ============================================================
def download_audio(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    outtmpl = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)

    return path.rsplit(".", 1)[0] + ".mp3"


# ============================================================
# IMPORTANT SEGMENTS
# ============================================================
IMPORTANT_KEYWORDS = [
    "step", "important", "remember", "key", "mistake",
    "note", "first", "second", "finally", "example",
    "diagram", "chart", "figure"
]


def extract_important_segments(segments: List[Dict], max_chars=3000) -> str:
    text = ""
    for s in segments:
        t = s["text"].strip()
        if len(t.split()) < 6:
            continue
        if any(k in t.lower() for k in IMPORTANT_KEYWORDS):
            text += t + " "
        if len(text) >= max_chars:
            break
    return text.strip()


# ============================================================
# FRAME EXTRACTION (UPLOADS ONLY)
# ============================================================
def extract_frame(video_path: str, timestamp: float, out_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
            "-y",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def extract_key_frames(
    segments: List[Dict],
    video_path: str,
    max_frames=3
) -> List[Dict]:
    frames = []

    for i, s in enumerate(segments):
        if any(k in s["text"].lower() for k in IMPORTANT_KEYWORDS):
            frame_path = f"frame_{i}.jpg"
            extract_frame(video_path, s["start"], frame_path)
            frames.append({
                "path": frame_path,
                "text": s["text"],
                "start": s["start"],
            })
        if len(frames) >= max_frames:
            break

    return frames


# ============================================================
# UI
# ============================================================
st.title("🎥 AI Video Understanding Tool")
st.caption("From long videos → transcripts → insights → visuals")

youtube_url = st.text_input("YouTube link (optional)")
uploaded_file = st.file_uploader(
    "Or upload a video/audio file",
    type=["mp4", "mp3", "wav", "m4a", "mov"]
)

max_frames = st.slider(
    "How many key screenshots to extract?",
    min_value=1,
    max_value=5,
    value=3
)

st.info(
    "📌 Screenshots are available only for uploaded video files. "
    "YouTube links are analyzed via audio only."
)


# ============================================================
# MAIN PIPELINE
# ============================================================
if st.button("🚀 Analyze Video"):

    if not youtube_url and not uploaded_file:
        st.error("Please provide a YouTube link or upload a file.")
        st.stop()

    try:
        # -------- SOURCE --------
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            is_video = suffix.lower() in [".mp4", ".mov", ".mkv"]
            audio_path = file_path
            video_path = file_path if is_video else None

        else:
            with st.spinner("Downloading audio from YouTube..."):
                audio_path = download_audio(youtube_url)
            video_path = None  # IMPORTANT

        # -------- TRANSCRIPTION --------
        with st.spinner("Transcribing with Whisper..."):
            model = load_whisper_model()
            segments, _ = model.transcribe(audio_path)

            segment_data = [{
                "text": s.text,
                "start": s.start,
                "end": s.end,
            } for s in segments]

        st.session_state.segments = segment_data

        # -------- IMPORTANT TEXT --------
        important_text = extract_important_segments(segment_data)
        st.session_state.raw_transcript = important_text

        # -------- FRAMES (ONLY IF VIDEO EXISTS) --------
        if video_path:
            with st.spinner("Extracting key screenshots..."):
                st.session_state.frames = extract_key_frames(
                    segment_data,
                    video_path,
                    max_frames=max_frames
                )
        else:
            st.session_state.frames = []

        # -------- REFINEMENT --------
        with st.spinner("Generating core understanding..."):
            st.session_state.refined_transcript = generate_text(
                refined_transcript_prompt(important_text)
            )

    except Exception as e:
        st.error("Unexpected error during processing.")
        st.exception(e)
        st.stop()


# ============================================================
# IMAGE EXPLANATIONS
# ============================================================
if st.session_state.frames:
    st.divider()
    st.subheader("🖼️ Key Visual Moments")

    for idx, frame in enumerate(st.session_state.frames, start=1):
        st.image(frame["path"], caption=f"Moment {idx}", use_column_width=True)

        with st.expander("Explain this image"):
            explanation = generate_text(
                f"""
This image comes from a video.

At this moment, the speaker is saying:
{frame['text']}

Explain what the image represents.
Explain it simply.
Explain why it matters.
Limit to 4–5 sentences.
"""
            )
            st.write(explanation)


# ============================================================
# INSIGHTS
# ============================================================
if st.session_state.refined_transcript:
    st.divider()
    st.subheader("🧠 What Actually Matters")
    st.write(st.session_state.refined_transcript)

    c1, c2, c3 = st.columns(3)

    if c1.button("💡 Key Insights"):
        st.session_state.takeaways = generate_text(
            key_takeaways_prompt(st.session_state.refined_transcript)
        )

    if c2.button("🚫 Common Mistakes"):
        st.session_state.mistakes = generate_text(
            mistakes_prompt(st.session_state.refined_transcript)
        )

    if c3.button("🛠️ Practical Application"):
        st.session_state.application = generate_text(
            application_prompt(st.session_state.refined_transcript)
        )


# ============================================================
# SOCIAL CONTENT
# ============================================================
if st.session_state.refined_transcript:
    st.divider()
    st.header("✍️ Repurpose as Content")

    if st.button("🐦 Twitter Thread"):
        st.session_state.twitter_content = generate_text(
            twitter_prompt(st.session_state.refined_transcript)
        )

    if st.button("💼 LinkedIn Post"):
        st.session_state.linkedin_content = generate_text(
            linkedin_prompt(st.session_state.refined_transcript)
        )

    if st.button("🎥 Reel Hooks"):
        st.session_state.reel_content = generate_text(
            reel_prompt(st.session_state.refined_transcript)
        )


# ============================================================
# RESET
# ============================================================
st.divider()
if st.button("🔄 Start Over"):
    st.session_state.clear()
