import os
import tempfile
import streamlit as st
import yt_dlp
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

# ======================
# SESSION STATE INIT
# ======================
for key in [
    "raw_transcript",
    "refined_transcript",
    "takeaways",
    "mistakes",
    "application",
    "twitter_content",
    "linkedin_content",
    "reel_content",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# ======================
# LOAD WHISPER ONCE (CRITICAL FOR SPEED)
# ======================
@st.cache_resource
def load_whisper_model():
    # Use "tiny" for maximum speed, "small" for balance
    return whisper.load_model("tiny")


# ======================
# UI – HEADER
# ======================
st.title("AI Content Repurposer")
st.caption("Turn long YouTube videos into fast, useful insights")
st.caption("Smart analysis → Insights → Content")

youtube_url = st.text_input("YouTube video link")

st.caption(
    "This tool analyzes only the most important parts of a video for speed. "
    "Works well even for long videos (20–60+ minutes)."
)


# ======================
# AUDIO DOWNLOAD
# ======================
def download_audio(url):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_path,
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 30,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)
        audio_file = audio_file.rsplit(".", 1)[0] + ".mp3"

    return audio_file


# ======================
# SMART SEGMENT EXTRACTION (KEY OPTIMIZATION)
# ======================
IMPORTANT_KEYWORDS = [
    "step", "important", "remember", "key", "mistake",
    "note", "first", "second", "finally", "example"
]

def extract_important_segments(segments, max_chars=3000):
    collected = ""

    for seg in segments:
        text = seg["text"].strip()

        # Skip very short filler segments
        if len(text.split()) < 6:
            continue

        # Keep high-signal segments
        if any(k in text.lower() for k in IMPORTANT_KEYWORDS):
            collected += text + " "

        if len(collected) >= max_chars:
            break

    return collected.strip()


# ======================
# TRANSCRIPTION PIPELINE (FAST + SCALABLE)
# ======================
if st.button("🎧 Analyze Video"):
    if not youtube_url:
        st.error("Please paste a YouTube link.")
    else:
        try:
            with st.spinner("Downloading audio..."):
                audio_path = download_audio(youtube_url)

            with st.spinner("Transcribing key segments only..."):
                model = load_whisper_model()
                result = model.transcribe(
                    audio_path,
                    fp16=False,
                    verbose=False
                )

            # 🔥 CORE SPEED TRICK
            important_text = extract_important_segments(result["segments"])
            st.session_state.raw_transcript = important_text

            with st.spinner("Extracting what actually matters..."):
                st.session_state.refined_transcript = generate_text(
                    refined_transcript_prompt(important_text)
                )

        except Exception as e:
            st.error("Something went wrong during processing.")
            st.write(e)


# ======================
# INSIGHTS (LAZY-LOADED)
# ======================
if st.session_state.refined_transcript:

    st.divider()
    st.subheader("🧠 What Actually Matters")
    st.write(st.session_state.refined_transcript)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💡 Key Insights"):
            st.session_state.takeaways = generate_text(
                key_takeaways_prompt(st.session_state.refined_transcript)
            )

    with col2:
        if st.button("🚫 Common Mistakes"):
            st.session_state.mistakes = generate_text(
                mistakes_prompt(st.session_state.refined_transcript)
            )

    with col3:
        if st.button("🛠️ Practical Application"):
            st.session_state.application = generate_text(
                application_prompt(st.session_state.refined_transcript)
            )

    if st.session_state.takeaways:
        st.subheader("💡 Key Insights")
        st.write(st.session_state.takeaways)

    if st.session_state.mistakes:
        st.subheader("🚫 Where People Go Wrong")
        st.write(st.session_state.mistakes)

    if st.session_state.application:
        st.subheader("🛠️ Practical Application")
        st.write(st.session_state.application)

    with st.expander("📄 View Extracted Transcript"):
        st.write(st.session_state.raw_transcript)


# ======================
# SOCIAL MEDIA CONTENT (ON DEMAND)
# ======================
if st.session_state.refined_transcript:

    st.divider()
    st.header("✍️ Turn This Into Content")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🐦 Twitter Thread"):
            st.session_state.twitter_content = generate_text(
                twitter_prompt(st.session_state.refined_transcript)
            )

    with col2:
        if st.button("💼 LinkedIn Post"):
            st.session_state.linkedin_content = generate_text(
                linkedin_prompt(st.session_state.refined_transcript)
            )

    with col3:
        if st.button("🎥 Reel Hooks"):
            st.session_state.reel_content = generate_text(
                reel_prompt(st.session_state.refined_transcript)
            )

    if st.session_state.twitter_content:
        st.subheader("🐦 Twitter/X Thread")
        st.text_area("", st.session_state.twitter_content, height=220)

    if st.session_state.linkedin_content:
        st.subheader("💼 LinkedIn Post")
        st.text_area("", st.session_state.linkedin_content, height=220)

    if st.session_state.reel_content:
        st.subheader("🎥 Reel Hooks")
        st.text_area("", st.session_state.reel_content, height=150)


# ======================
# RESET
# ======================
st.divider()
if st.button("🔄 Start Over"):
    for key in st.session_state:
        st.session_state[key] = None

st.caption("Optimized for long videos, fast insights, and real creators.")

