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
    st.session_state.setdefault(key, None)


# ======================
# LOAD WHISPER
# ======================
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# ======================
# DOWNLOAD AUDIO
# ======================
def download_audio(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        return file_path.rsplit(".", 1)[0] + ".mp3"


# ======================
# IMPORTANT SEGMENTS
# ======================
IMPORTANT_KEYWORDS = [
    "step", "important", "remember", "key", "mistake",
    "note", "first", "second", "finally", "example"
]

def extract_important_segments(segments, max_chars=3000):
    collected = ""
    for seg in segments:
        text = seg["text"].strip()
        if len(text.split()) < 6:
            continue
        if any(k in text.lower() for k in IMPORTANT_KEYWORDS):
            collected += text + " "
        if len(collected) >= max_chars:
            break
    return collected.strip()


# ======================
# UI
# ======================
st.title("AI Content Repurposer")
st.caption("Turn long videos into fast, useful insights")

youtube_url = st.text_input("YouTube video link (optional)")
uploaded_file = st.file_uploader(
    "Or upload a video/audio file",
    type=["mp4", "mp3", "wav", "m4a", "mov"]
)

st.info(
    "If a YouTube video can’t be downloaded due to restrictions, "
    "upload the video or audio file directly."
)


# ======================
# MAIN PIPELINE (ONE BUTTON)
# ======================
if st.button("🎧 Analyze Video"):

    if not youtube_url and not uploaded_file:
        st.error("Please provide a YouTube link or upload a file.")
        st.stop()

    try:
        # -------- SOURCE --------
        if uploaded_file:
            with st.spinner("Processing uploaded file..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    audio_path = tmp.name
        else:
            with st.spinner("Downloading audio from YouTube..."):
                audio_path = download_audio(youtube_url)

        # -------- TRANSCRIBE --------
        with st.spinner("Transcribing key segments..."):
            model = load_whisper_model()
            segments, _ = model.transcribe(audio_path)
            result = {"segments": [{"text": s.text} for s in segments]}

        important_text = extract_important_segments(result["segments"])
        st.session_state.raw_transcript = important_text

        with st.spinner("Extracting what matters..."):
            st.session_state.refined_transcript = generate_text(
                refined_transcript_prompt(important_text)
            )

    except Exception:
        st.warning(
            "🚫 This video can’t be processed automatically.\n\n"
            "👉 Please upload the video or audio file instead."
        )
        st.stop()


# ======================
# INSIGHTS
# ======================
if st.session_state.refined_transcript:
    st.divider()
    st.subheader("🧠 What Actually Matters")
    st.write(st.session_state.refined_transcript)

    col1, col2, col3 = st.columns(3)

    if col1.button("💡 Key Insights"):
        st.session_state.takeaways = generate_text(
            key_takeaways_prompt(st.session_state.refined_transcript)
        )

    if col2.button("🚫 Common Mistakes"):
        st.session_state.mistakes = generate_text(
            mistakes_prompt(st.session_state.refined_transcript)
        )

    if col3.button("🛠️ Practical Application"):
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


# ======================
# SOCIAL CONTENT
# ======================
if st.session_state.refined_transcript:
    st.divider()
    st.header("✍️ Turn This Into Content")

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

    if st.session_state.twitter_content:
        st.text_area("Twitter/X", st.session_state.twitter_content, height=220)

    if st.session_state.linkedin_content:
        st.text_area("LinkedIn", st.session_state.linkedin_content, height=220)

    if st.session_state.reel_content:
        st.text_area("Reels", st.session_state.reel_content, height=150)


# ======================
# RESET
# ======================
st.divider()
if st.button("🔄 Start Over"):
    st.session_state.clear()
