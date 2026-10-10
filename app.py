import streamlit as st
from openai import OpenAI
import PyPDF2
import fitz  # PyMuPDF
import json
import io
import re
import base64
import os
import textwrap
from dotenv import load_dotenv
from ddgs import DDGS

from database import init_db, add_subject, get_all_subjects, get_subject, delete_subject

load_dotenv()
init_db()

# 1. Initialize clients.
# 'client' talks to OpenRouter (used for NVIDIA Nemotron models, text + vision).
# 'groq_client' talks to Groq, a separate service, used only as a last-resort
# fallback for vision/OCR if OpenRouter's NVIDIA vision endpoints are down.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

TEXT_CANDIDATES = [
    (client, "nvidia/nemotron-3-super-120b-a12b:free"),
    (client, "nvidia/nemotron-nano-9b-v2:free"),
]
VISION_CANDIDATES = [
    (client, "nvidia/nemotron-nano-12b-v2-vl:free"),   # NVIDIA, free tier
    (client, "nvidia/nemotron-nano-12b-v2-vl"),         # NVIDIA, paid tier (needs OpenRouter credit)
    (groq_client, "qwen/qwen3.8-27b"),                  # Groq, separate service, last resort
]

NO_THINKING_INSTRUCTION = "detailed thinking off"

# How much of an uploaded file we keep. Raised from the original defaults so
# larger documents (like a full book) are covered more completely. The
# tradeoff: a bigger vision_pages value means a slower "Save Subject" step
# for scanned/image-based files, since each page is its own AI call.
MAX_CHARS = 20000
MAX_TEXT_PAGES = 60
MAX_VISION_PAGES = 15


def strip_reasoning(text):
    """Removes any <think>...</think> reasoning trace the model might still include."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat_completion(messages, temperature=0.4, candidates=None, retries=1):
    """Calls the chat API, trying each (client, model) pair in 'candidates' in
    order, retrying each one before moving to the next. Raises the last error
    if everything fails."""
    candidates = candidates or TEXT_CANDIDATES
    last_error = None
    for c, model in candidates:
        for attempt in range(retries):
            try:
                return c.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
            except Exception as e:
                last_error = e
                continue
    raise last_error


st.set_page_config(page_title="StudySpace", page_icon="📚", layout="centered")

# Sticky header: a marker div is placed as the first thing inside a container;
# this CSS makes the nearest block that contains that marker stick to the top
# of the page while everything else scrolls underneath it. This technique is
# more broadly compatible across Streamlit versions than relying on container
# 'key' class names alone.
st.markdown(
    textwrap.dedent("""
    <style>
    /* ---------- Visual polish: give the app a cleaner, modern chat-app look ---------- */

    /* Sidebar: slightly darker panel, clear separation from main content */
    section[data-testid="stSidebar"] {
        background-color: #16181d;
        border-right: 1px solid #262a33;
    }
    section[data-testid="stSidebar"] .stButton button {
        text-align: left;
        border-radius: 10px;
    }

    /* Chat bubbles, closer to ChatGPT/Claude/Gemini style: soft rounded
       panels with a subtle border instead of plain text in a flat list */
    div[data-testid="stChatMessage"] {
        background-color: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }

    /* Chat input box: rounded, slightly larger, easier to spot at a glance */
    div[data-testid="stChatInput"] {
        border-radius: 16px;
    }

    /* Buttons generally: rounded corners feel less "form-like", more "app-like" */
    .stButton button, .stDownloadButton button {
        border-radius: 10px;
    }

    /* Text inputs / text areas: match the rounded theme */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
    }

    /* Tighter, calmer heading weight so the page doesn't feel like a form */
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; }
    </style>
    """),
    unsafe_allow_html=True,
)


def start_sticky_header():
    """Pinning the header didn't hold up reliably across attempts, so this
    is now a no-op — the header renders normally, in flow, like the rest
    of the page."""
    pass


def header_spacer(rem):
    pass


# ---------- Session state ----------
if "active_subject_id" not in st.session_state:
    st.session_state.active_subject_id = None
if "show_create_form" not in st.session_state:
    st.session_state.show_create_form = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "focus_mode" not in st.session_state:
    st.session_state.focus_mode = False
if "muted" not in st.session_state:
    st.session_state.muted = False
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "learn"  # "learn" or "test"


# ---------- Vision + web search helpers ----------

def describe_image_bytes(image_bytes, mime_type, prompt):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": NO_THINKING_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
            ]
        }
    ]
    response = chat_completion(messages, temperature=0.2, candidates=VISION_CANDIDATES, retries=1)
    return strip_reasoning(response.choices[0].message.content)


def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        formatted = ""
        for r in results:
            formatted += f"- {r.get('title', '')}: {r.get('body', '')} (Source: {r.get('href', '')})\n"
        return formatted
    except Exception:
        return ""


def extract_text(file, max_chars=MAX_CHARS, max_pages=MAX_TEXT_PAGES, vision_pages=MAX_VISION_PAGES):
    """Reads text out of a TXT, PDF, or image file. Falls back to the vision
    model for images and for PDFs that turn out to be scanned (no real text).
    Returns (text, warning) — warning is None unless something worth flagging
    to the user happened during extraction."""

    if file.type == "text/plain":
        return file.read().decode("utf-8")[:max_chars], None

    elif file.type in ("image/png", "image/jpeg", "image/jpg"):
        image_bytes = file.read()
        try:
            description = describe_image_bytes(
                image_bytes, file.type,
                "Transcribe all readable text from this image exactly. "
                "If there are diagrams or charts, briefly describe what they show."
            )
            return description[:max_chars], None
        except Exception as e:
            return "", f"Couldn't read this image (AI vision service error: {e})"

    elif file.type == "application/pdf":
        file_bytes = file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page_num, page in enumerate(reader.pages):
            if page_num >= max_pages or len(text) >= max_chars:
                break
            text += page.extract_text() or ""

        if len(text.strip()) < 50:
            # Likely a scanned/image-based PDF — read it with the vision model instead.
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            vision_text = ""
            failed_pages = 0
            pages_to_try = min(len(doc), vision_pages)
            for page_num in range(pages_to_try):
                pix = doc[page_num].get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                try:
                    page_text = describe_image_bytes(
                        img_bytes, "image/png",
                        "Transcribe all readable text from this page exactly. "
                        "If there are diagrams or charts, briefly describe what they show."
                    )
                except Exception as e:
                    print(f"Page {page_num} failed: {e}")
                    page_text = ""
                    failed_pages += 1
                vision_text += page_text + "\n"
                if len(vision_text) >= max_chars:
                    break
            doc.close()

            if len(vision_text.strip()) < 20:
                return "", (
                    "This looks like a scanned/image-based PDF, and none of the "
                    "AI vision services (NVIDIA free, NVIDIA paid, or Groq) could "
                    "read it right now. Try again in a few minutes, or upload a "
                    "smaller/clearer file."
                )
            warning = None
            if failed_pages:
                warning = f"Note: {failed_pages} of {pages_to_try} pages couldn't be read and were skipped."
            return vision_text[:max_chars], warning

        return text[:max_chars], None

    return "", "Unsupported file type."


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F"
    "\U0000200D"
    "]+", flags=re.UNICODE
)


def clean_for_speech(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)        # code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)                       # inline code
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)     # headings
    text = re.sub(r"^[\s\-\*\|:]+$", "", text, flags=re.MULTILINE)  # table separators / horizontal rules
    text = text.replace("|", " ")                                  # table pipes
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)                 # bold
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)                    # italics
    text = re.sub(r"^[\s]*[-\*\+]\s+", "", text, flags=re.MULTILINE)  # bullet points
    text = EMOJI_PATTERN.sub("", text)                              # emojis
    text = text.replace('"', "")
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


if "speak_counter" not in st.session_state:
    st.session_state.speak_counter = 0


def speak(text):
    """Speaks 'text' out loud, with its own Pause/Resume buttons embedded in
    the same HTML block. Keeping the buttons in the same block as the audio
    (instead of separate st.button() widgets) means clicking them doesn't
    trigger a Streamlit rerun, which would otherwise tear down and rebuild
    the audio player and break pause/resume."""
    if st.session_state.muted:
        return
    clean_text = clean_for_speech(text)
    if not clean_text:
        return

    st.session_state.speak_counter += 1
    control_id = st.session_state.speak_counter
    safe_text = clean_text.replace("\\", "\\\\").replace('"', '\\"')

    html = f"""
    <div style="font-family: sans-serif;">
        <button id="pause-{control_id}" style="margin-right:6px;">⏸️ Pause</button>
        <button id="resume-{control_id}">▶️ Resume</button>
    </div>
    <script>
        window.speechSynthesis.cancel();
        var utter_{control_id} = new SpeechSynthesisUtterance("{safe_text}");
        window.speechSynthesis.speak(utter_{control_id});

        document.getElementById("pause-{control_id}").onclick = function() {{
            window.speechSynthesis.pause();
        }};
        document.getElementById("resume-{control_id}").onclick = function() {{
            window.speechSynthesis.resume();
        }};
    </script>
    """
    st.components.v1.html(html, height=40)


def apply_mute_state():
    if st.session_state.muted:
        st.components.v1.html(
            "<script>window.speechSynthesis.cancel();</script>", height=0
        )


# ================= SIDEBAR: Subject list + New Subject button (hidden in Focus Mode) =================
if not st.session_state.focus_mode:
    with st.sidebar:
        st.subheader("📚 Your Subjects")

        subjects = get_all_subjects()

        if subjects:
            for subject_id, subject_name in subjects:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(subject_name, key=f"subject_{subject_id}", use_container_width=True):
                        st.session_state.active_subject_id = subject_id
                        st.session_state.show_create_form = False
                        st.session_state.questions = []
                        st.session_state.current_q = 0
                        st.session_state.feedback = None
                        st.session_state.view_mode = "learn"
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{subject_id}"):
                        delete_subject(subject_id)
                        if st.session_state.active_subject_id == subject_id:
                            st.session_state.active_subject_id = None
                        st.rerun()
        else:
            st.caption("No subjects yet.")

        st.divider()
        if st.button("➕ New Subject", use_container_width=True):
            st.session_state.active_subject_id = None
            st.session_state.show_create_form = True
            st.rerun()


# ================= MAIN AREA =================
if st.session_state.show_create_form:
    with st.container():
        start_sticky_header()
        st.title("📚 StudySpace")
        st.checkbox("🔇 Mute", key="muted")
        apply_mute_state()
    header_spacer(5)

    st.subheader("➕ Create a New Subject")

    new_subject_name = st.text_input("Subject name (e.g. 'Biology Chapter 3')")
    uploaded_file = st.file_uploader(
        "Upload study notes (PDF, TXT, or an image of your notes)",
        type=["pdf", "txt", "png", "jpg", "jpeg"]
    )

    if st.button("Save Subject"):
        if new_subject_name and uploaded_file:
            with st.spinner("Reading your notes... this can take a while for large scanned files."):
                notes_text, warning = extract_text(uploaded_file)
            if notes_text.strip():
                add_subject(new_subject_name, notes_text)
                st.session_state.show_create_form = False
                if warning:
                    st.warning(warning)
                st.success(f"Saved '{new_subject_name}'!")
                st.rerun()
            else:
                st.error(warning or "Couldn't read any content from that file. Try a different file.")
        else:
            st.warning("Please give it a name and upload a file first.")

elif st.session_state.active_subject_id is None:
    with st.container():
        start_sticky_header()
        st.title("📚 StudySpace")
        st.checkbox("🔇 Mute", key="muted")
        apply_mute_state()
    header_spacer(5)

    st.info("Pick a subject from the sidebar, or click '➕ New Subject' to create one.")

else:
    subject_id, subject_name, notes_text = get_subject(st.session_state.active_subject_id)

    with st.container():
        start_sticky_header()

        if st.session_state.focus_mode:
            _, mute_col = st.columns([4, 1])
            with mute_col:
                st.checkbox("🔇 Mute", key="muted")
                apply_mute_state()
        else:
            title_col, mute_col = st.columns([4, 1])
            with title_col:
                st.title("📚 StudySpace")
            with mute_col:
                st.checkbox("🔇 Mute", key="muted")
                apply_mute_state()

        nav_col, focus_col = st.columns([3, 1])
        with nav_col:
            st.subheader(f"📖 {subject_name}")
        with focus_col:
            st.toggle("Focus Mode", key="focus_mode")

        mode_col1, mode_col2 = st.columns([1, 1])
        with mode_col1:
            if st.button("💬 Learn", use_container_width=True,
                         type="primary" if st.session_state.view_mode == "learn" else "secondary"):
                st.session_state.view_mode = "learn"
                st.rerun()
        with mode_col2:
            if st.button("📝 Test Me", use_container_width=True,
                         type="primary" if st.session_state.view_mode == "test" else "secondary"):
                st.session_state.view_mode = "test"
                st.rerun()
    header_spacer(11)

    # ---------- LEARN MODE ----------
    if st.session_state.view_mode == "learn":
        if subject_id not in st.session_state.chat_history:
            st.session_state.chat_history[subject_id] = []

        use_web_search = st.checkbox("🌐 Also search the web for extra context", key="use_web_search")

        for msg in st.session_state.chat_history[subject_id]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Placed at the top level (not inside any container/columns) so Streamlit
        # pins it to the bottom of the page automatically, like a normal chat app.
        user_question = st.chat_input("Ask about your notes...")

        if user_question:
            st.session_state.chat_history[subject_id].append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    system_prompt = (
                        f"{NO_THINKING_INSTRUCTION}\n\n"
                        "You are a patient, encouraging study tutor. Prioritize the "
                        "student's notes below when answering. Explain clearly and "
                        "simply, and offer examples when helpful. If something isn't "
                        "covered by the notes, say plainly that it isn't in the notes "
                        "provided, rather than saying you can't access files.\n\n"
                        f"STUDENT'S NOTES:\n{notes_text}"
                    )

                    if use_web_search:
                        with st.spinner("Searching the web..."):
                            web_results = web_search(user_question)
                        if web_results:
                            system_prompt += (
                                "\n\nADDITIONAL WEB SEARCH RESULTS (use alongside the "
                                "notes; make clear when you're drawing on these instead "
                                f"of the notes):\n{web_results}"
                            )

                    history_messages = [{"role": "system", "content": system_prompt}]
                    history_messages += st.session_state.chat_history[subject_id]

                    try:
                        response = chat_completion(history_messages, temperature=0.4)
                        answer = strip_reasoning(response.choices[0].message.content)
                    except Exception as e:
                        answer = f"Sorry, the AI service didn't respond. ({e})"

                    st.markdown(answer)
                    speak(answer)

            st.session_state.chat_history[subject_id].append(
                {"role": "assistant", "content": answer}
            )

    # ---------- TEST MODE ----------
    else:
        if st.button("Generate Exam Session"):
            with st.spinner("Analyzing notes with AI..."):
                prompt = f"""
                Extract 3 key exam questions from these study notes.
                Format the response strictly as a JSON array of objects with keys "question" and "answer".
                Notes: {notes_text}
                """

                try:
                    response = chat_completion(
                        [
                            {"role": "system", "content": f"{NO_THINKING_INSTRUCTION}\nYou are an expert exam tutor. Respond strictly in JSON format, with nothing else."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )

                    raw_content = strip_reasoning(response.choices[0].message.content).strip()

                    if raw_content.startswith("```"):
                        raw_content = raw_content.strip("`")
                        raw_content = raw_content.replace("json", "", 1).strip()

                    parsed = json.loads(raw_content)

                    if isinstance(parsed, list):
                        question_list = parsed
                    elif isinstance(parsed, dict):
                        question_list = next((v for v in parsed.values() if isinstance(v, list)), [])
                    else:
                        question_list = []

                    if not question_list:
                        st.error("The AI didn't return any questions. Try clicking Generate again.")
                    else:
                        st.session_state.questions = question_list
                        st.session_state.current_q = 0
                        st.session_state.feedback = None
                        st.success("Quiz generated successfully!")

                except Exception as e:
                    st.error(f"Couldn't generate the quiz right now. ({e})")

        if st.session_state.questions:
            q_index = st.session_state.current_q
            if q_index < len(st.session_state.questions):
                current = st.session_state.questions[q_index]

                st.subheader(f"Question {q_index + 1} of {len(st.session_state.questions)}")
                st.markdown(f"**{current['question']}**")

                speak(current["question"])

                user_answer = st.text_input(
                    "Your Answer (or speak into text input):", key=f"ans_{q_index}"
                )

                if st.button("Submit Answer"):
                    eval_prompt = (
                        f"Question: {current['question']}\n"
                        f"Correct Answer: {current['answer']}\n"
                        f"Student Answer: {user_answer}\n"
                        "Grade the student answer (1-10) and give 1 brief feedback sentence."
                    )

                    try:
                        eval_response = chat_completion(
                            [
                                {"role": "system", "content": NO_THINKING_INSTRUCTION},
                                {"role": "user", "content": eval_prompt}
                            ],
                            temperature=0.0
                        )
                        st.session_state.feedback = strip_reasoning(eval_response.choices[0].message.content)
                    except Exception as e:
                        st.session_state.feedback = f"Couldn't grade that right now. ({e})"

                if st.session_state.feedback:
                    st.info(st.session_state.feedback)
                    speak(st.session_state.feedback)

                if st.button("Next Question"):
                    st.session_state.current_q += 1
                    st.session_state.feedback = None
                    st.rerun()

            else:
                st.balloons()
                st.success("Exam Session Completed!")