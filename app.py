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
    "image_explanations",
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
# YOUTUBE AUDIO (BEST-EFFORT ONLY)
# ============================================================
def download_audio(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    outtmpl = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "retries": 2,
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
# FRAME EXTRACTION (VIDEO FILES ONLY)
# ============================================================
def extract_frame(video_path: str, timestamp: float, out_path: str):
    subprocess.run(
        [
            "ffmpeg", "-ss", str(timestamp), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path, "-y"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def extract_key_frames(segments, video_path, max_frames=3):
    frames = []
    for i, s in enumerate(segments):
        if any(k in s["text"].lower() for k in IMPORTANT_KEYWORDS):
            frame = f"frame_{i}.jpg"
            extract_frame(video_path, s["start"], frame)
            frames.append({"path": frame, "text": s["text"]})
        if len(frames) >= max_frames:
            break
    return frames

# ============================================================
# UI
# ============================================================
st.title("🎥 AI Video & Image Understanding Tool")

youtube_url = st.text_input("YouTube link (best-effort)")
uploaded_video = st.file_uploader(
    "Upload video/audio/subtitles",
    type=["mp4", "mp3", "wav", "m4a", "srt", "vtt"]
)

uploaded_images = st.file_uploader(
    "Upload image(s) to explain",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

max_frames = st.slider("Screenshots from video", 1, 5, 3)

with st.expander("⚠️ If YouTube download fails"):
    st.markdown("""
Use a manual downloader (e.g. y2mate or VLC),
then upload the MP4 / MP3 / subtitle file here.
""")

# ============================================================
# MAIN PIPELINE
# ============================================================
if st.button("🚀 Analyze"):

    try:
        segments = []

        # -------- SUBTITLES UPLOAD --------
        if uploaded_video and uploaded_video.name.endswith((".srt", ".vtt")):
            text = uploaded_video.read().decode("utf-8", errors="ignore")
            st.session_state.raw_transcript = text

        # -------- VIDEO / AUDIO UPLOAD --------
        elif uploaded_video:
            suffix = os.path.splitext(uploaded_video.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_video.read())
                path = tmp.name

            model = load_whisper_model()
            segs, _ = model.transcribe(path)
            segments = [{"text": s.text, "start": s.start} for s in segs]

            st.session_state.raw_transcript = extract_important_segments(segments)

            if suffix.lower() in [".mp4", ".mov", ".mkv"]:
                st.session_state.frames = extract_key_frames(
                    segments, path, max_frames
                )

        # -------- YOUTUBE --------
        elif youtube_url:
            audio_path = download_audio(youtube_url)
            model = load_whisper_model()
            segs, _ = model.transcribe(audio_path)
            segments = [{"text": s.text, "start": s.start} for s in segs]
            st.session_state.raw_transcript = extract_important_segments(segments)

        # -------- REFINEMENT --------
        st.session_state.refined_transcript = generate_text(
            refined_transcript_prompt(st.session_state.raw_transcript)
        )

    except Exception as e:
        st.error("Processing failed. Please upload files manually.")
        st.exception(e)

# ============================================================
# IMAGE EXPLANATION (UPLOADED IMAGES)
# ============================================================
if uploaded_images:
    st.divider()
    st.subheader("🖼️ Image Explanations")

    for img in uploaded_images:
        st.image(img)
        explanation = generate_text(
            "Explain this image in simple terms for a beginner."
        )
        st.write(explanation)

# ============================================================
# VIDEO FRAMES
# ============================================================
if st.session_state.frames:
    st.divider()
    st.subheader("🎞️ Video Screenshots")

    for f in st.session_state.frames:
        st.image(f["path"])
        st.write(generate_text(
            f"Explain this image based on context: {f['text']}"
        ))

# ============================================================
# INSIGHTS
# ============================================================
if st.session_state.refined_transcript:
    st.divider()
    st.write(st.session_state.refined_transcript)

# ============================================================
# RESET
# ============================================================
st.divider()
if st.button("🔄 Reset"):
    st.session_state.clear()
