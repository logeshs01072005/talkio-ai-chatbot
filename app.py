import streamlit as st
from google import genai
from google.genai import types
import requests
import hashlib
import time
import os
import base64
from PIL import Image
import io
from pypdf import PdfReader
from dotenv import load_dotenv
from auth import login_page
from rag_engine import HybridRAGEngine

load_dotenv()

# -----------------------------
# Config
# -----------------------------
API_URL     = "http://127.0.0.1:8000"

# Works locally (.env) AND on Streamlit Cloud (secrets)
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = os.getenv("API_KEY", "")

client      = genai.Client(api_key=API_KEY)
MODEL       = "gemini-2.5-flash-lite"
TEMPERATURE = 0.3

# -----------------------------
# Current Affairs Auto-Detection
# -----------------------------
CURRENT_AFFAIRS_KEYWORDS = [
    "current", "now", "today", "latest", "recent", "2026", "2025",
    "who is", "present", "cm", "chief minister", "minister", "president",
    "prime minister", "ceo", "chairman", "winner", "election", "score",
    "price", "stock", "news", "update", "won", "lost", "result",
    "governor", "mayor", "leader", "appointed", "match", "live"
]

def is_current_affairs(query: str) -> bool:
    return any(kw in query.lower() for kw in CURRENT_AFFAIRS_KEYWORDS)

# -----------------------------
# Auto Title Generator
# -----------------------------
def generate_title(prompt: str) -> str:
    stopwords = ["what is", "what are", "who is", "who are", "how to",
                 "how do", "can you", "tell me", "explain", "describe",
                 "give me", "i want", "please", "help me", "about"]
    title = prompt.strip()
    for sw in stopwords:
        if title.lower().startswith(sw):
            title = title[len(sw):].strip()
            break
    title = title.capitalize()
    if len(title) > 40:
        title = title[:40].rsplit(" ", 1)[0] + "..."
    return title if title else prompt[:40]

# -----------------------------
# Image Helper
# -----------------------------
def image_to_base64(image_file) -> str:
    """Convert uploaded image to base64"""
    bytes_data = image_file.read()
    return base64.b64encode(bytes_data).decode("utf-8")

def get_image_mime(image_file) -> str:
    """Get MIME type from image file"""
    name = image_file.name.lower()
    if name.endswith(".png"):
        return "image/png"
    elif name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    elif name.endswith(".gif"):
        return "image/gif"
    elif name.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"

# -----------------------------
# API Helpers
# -----------------------------
def api_create_chat(user_id, title="New Chat"):
    try:
        r = requests.post(f"{API_URL}/chats", json={"user_id": user_id, "title": title})
        return r.json() if r.status_code == 201 else None
    except:
        return None

def api_update_chat_title(chat_id, title):
    try:
        requests.put(f"{API_URL}/chats/{chat_id}", json={"title": title})
    except:
        pass

def api_save_message(chat_id, role, content):
    try:
        r = requests.post(f"{API_URL}/messages",
                          json={"chat_id": chat_id, "role": role, "content": content})
        return r.json() if r.status_code == 201 else None
    except:
        return None

def api_get_user_chats(user_id):
    try:
        r = requests.get(f"{API_URL}/chats/user/{user_id}")
        return r.json() if r.status_code == 200 else []
    except:
        return []

def api_get_messages(chat_id):
    try:
        r = requests.get(f"{API_URL}/messages/chat/{chat_id}")
        return r.json() if r.status_code == 200 else []
    except:
        return []

def api_get_user(user_id):
    try:
        r = requests.get(f"{API_URL}/users/{user_id}")
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_delete_messages(chat_id):
    try:
        requests.delete(f"{API_URL}/messages/chat/{chat_id}")
    except:
        pass

def api_delete_chat(chat_id):
    try:
        requests.delete(f"{API_URL}/chats/{chat_id}")
    except:
        pass

# -----------------------------
# Gemini Generate (text)
# -----------------------------
def ask_gemini(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < 2:
                st.toast(f"⏳ Rate limit — retrying... ({attempt+1}/3)", icon="⚠️")
                time.sleep(30)
            elif "503" in error_str and attempt < 2:
                st.toast(f"⏳ Model busy — retrying... ({attempt+1}/3)", icon="⚠️")
                time.sleep(3)
            else:
                raise e

# -----------------------------
# Gemini Generate (image + text)
# -----------------------------
def ask_gemini_with_image(prompt: str, image_b64: str, mime_type: str) -> str:
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=base64.b64decode(image_b64),
                        mime_type=mime_type
                    ),
                    types.Part.from_text(text=prompt)
                ],
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < 2:
                st.toast(f"⏳ Rate limit — retrying... ({attempt+1}/3)", icon="⚠️")
                time.sleep(30)
            elif "503" in error_str and attempt < 2:
                st.toast(f"⏳ Model busy — retrying... ({attempt+1}/3)", icon="⚠️")
                time.sleep(3)
            else:
                raise e

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Talkio AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION PERSISTENCE — restore login after refresh
# =====================================================
def restore_session():
    """
    Restore session from query params or st.session_state
    Uses st.query_params to persist user_id across refresh
    """
    params = st.query_params

    # If user_id in URL params → restore session
    if "uid" in params and not st.session_state.get("logged_in"):
        user_id = params["uid"]
        user    = api_get_user(user_id)

        if user:
            st.session_state.logged_in = True
            st.session_state.username  = user["username"]
            st.session_state.user_id   = user_id

            # Restore last chat
            if "cid" in params:
                st.session_state.chat_id = params["cid"]
                raw_messages = api_get_messages(params["cid"])
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in raw_messages
                ]
                st.session_state.title_set = True

restore_session()

# -----------------------------
# LOGIN CHECK
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Save user_id + chat_id to URL so refresh restores session
if st.session_state.get("user_id"):
    params = {"uid": st.session_state.user_id}
    if st.session_state.get("chat_id"):
        params["cid"] = st.session_state.chat_id
    st.query_params.update(params)

# -----------------------------
# RAG Engine
# -----------------------------
@st.cache_resource
def get_rag_engine():
    return HybridRAGEngine()

rag = get_rag_engine()

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = {}
if "title_set" not in st.session_state:
    st.session_state.title_set = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "image_mime" not in st.session_state:
    st.session_state.image_mime = None

# -----------------------------
# Load Chat Function
# -----------------------------
def load_chat(chat_id: str, chat_title: str):
    raw_messages = api_get_messages(chat_id)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in raw_messages
    ]
    st.session_state.chat_id   = chat_id
    st.session_state.title_set = True
    st.session_state.uploaded_image = None
    st.query_params.update({
        "uid": st.session_state.user_id,
        "cid": chat_id
    })
    st.toast(f"✅ Loaded: {chat_title[:25]}", icon="💬")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🤖 Talkio AI")
    st.markdown("---")

    username = st.session_state.get("username", "User")
    st.success(f"👤 **{username}**")

    if st.button("🚪 Logout", use_container_width=True):
        st.query_params.clear()
        for key in ["logged_in","username","messages","chat_id",
                    "doc_ids","user_id","title_set","uploaded_image","image_mime"]:
            st.session_state[key] = (
                False if key in ["logged_in","title_set"]
                else [] if key == "messages"
                else {} if key == "doc_ids"
                else None if key in ["chat_id","uploaded_image","image_mime"]
                else ""
            )
        st.rerun()

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.chat_id        = None
        st.session_state.title_set      = False
        st.session_state.uploaded_image = None
        params = {"uid": st.session_state.user_id}
        st.query_params.update(params)
        st.rerun()

    st.markdown("---")

    use_web = st.toggle("🌐 Web Search", value=False)

    st.markdown("---")

    # PDF Upload
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if rag.get_loaded_docs():
        st.subheader("📚 Loaded")
        for doc in rag.get_loaded_docs():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {doc['doc_name'][:22]}")
            with col2:
                if st.button("❌", key=f"rm_{doc['doc_id']}"):
                    rag.remove_document(doc["doc_id"])
                    st.rerun()

    st.markdown("---")

    # Chat History
    st.subheader("💬 Chat History")
    if st.session_state.get("user_id"):
        user_chats = api_get_user_chats(st.session_state.user_id)
        if user_chats:
            for chat in user_chats:
                col1, col2 = st.columns([4, 1])
                with col1:
                    is_active = chat["id"] == st.session_state.chat_id
                    icon      = "✅" if is_active else "💬"
                    label     = f"{icon} {chat['title'][:25]}"
                    if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                        load_chat(chat["id"], chat["title"])
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{chat['id']}"):
                        if st.session_state.chat_id == chat["id"]:
                            st.session_state.messages  = []
                            st.session_state.chat_id   = None
                            st.session_state.title_set = False
                        api_delete_chat(chat["id"])
                        st.toast("Chat deleted!", icon="🗑️")
                        st.rerun()
        else:
            st.caption("No chat history yet")

    st.markdown("---")
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.title_set = False
        if st.session_state.get("chat_id"):
            api_delete_messages(st.session_state.chat_id)
        st.rerun()

# -----------------------------
# Auto Create New Chat
# -----------------------------
if st.session_state.get("user_id") and st.session_state.chat_id is None:
    chat = api_create_chat(
        user_id=st.session_state.user_id,
        title="New Chat"
    )
    if chat:
        st.session_state.chat_id   = chat["id"]
        st.session_state.title_set = False
        st.query_params.update({
            "uid": st.session_state.user_id,
            "cid": chat["id"]
        })

# -----------------------------
# PDF Helpers
# -----------------------------
def read_pdf(pdf) -> str:
    text = ""
    reader = PdfReader(pdf)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text, chunk_size=1500, overlap=300) -> list:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def doc_hash(name, size) -> str:
    return hashlib.md5(f"{name}_{size}".encode()).hexdigest()

# -----------------------------
# Process PDFs
# -----------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = doc_hash(uploaded_file.name, uploaded_file.size)
        if file_id not in st.session_state.doc_ids:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                pdf_text = read_pdf(uploaded_file)
                chunks   = chunk_text(pdf_text)
                result   = rag.load_document(
                    chunks, doc_id=file_id, doc_name=uploaded_file.name
                )
                st.session_state.doc_ids[uploaded_file.name] = file_id
            st.toast(f"✅ {uploaded_file.name} indexed!", icon="📄")

# -----------------------------
# Main Area
# -----------------------------
st.title("🤖 Talkio AI")

if rag.is_loaded:
    docs = " · ".join([f"📄 {d['doc_name']}" for d in rag.get_loaded_docs()])
    st.info(f"📚 {docs}")

if not st.session_state.messages:
    st.markdown("### ✨ How can I help you today?")
    st.caption("Ask anything, upload PDFs, or attach an image to analyze.")
    st.markdown("---")

# -----------------------------
# Show Chat Messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show image if attached to message
        if msg.get("image_b64"):
            img_bytes = base64.b64decode(msg["image_b64"])
            img       = Image.open(io.BytesIO(img_bytes))
            st.image(img, width=300)
        st.write(msg["content"])

# -----------------------------
# Image Upload (above chat input)
# -----------------------------
st.markdown("**📸 Attach Image (optional)**")
uploaded_image = st.file_uploader(
    "Upload image to analyze",
    type=["png", "jpg", "jpeg", "gif", "webp"],
    key="img_uploader",
    label_visibility="collapsed"
)

if uploaded_image:
    # Preview image
    img = Image.open(uploaded_image)
    st.image(img, width=200, caption="Image attached ✅")
    uploaded_image.seek(0)
    st.session_state.uploaded_image = image_to_base64(uploaded_image)
    st.session_state.image_mime     = get_image_mime(uploaded_image)
else:
    if "img_uploader" not in st.session_state:
        st.session_state.uploaded_image = None
        st.session_state.image_mime     = None

# -----------------------------
# User Input
# -----------------------------
prompt = st.chat_input("Message Talkio AI...")

if prompt:

    # Auto-set title from FIRST message
    if not st.session_state.title_set and st.session_state.chat_id:
        smart_title = generate_title(prompt)
        api_update_chat_title(st.session_state.chat_id, smart_title)
        st.session_state.title_set = True

    # Capture image before resetting
    current_image     = st.session_state.uploaded_image
    current_image_mime = st.session_state.image_mime

    # Add user message
    user_msg = {"role": "user", "content": prompt}
    if current_image:
        user_msg["image_b64"] = current_image
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        if current_image:
            img_bytes = base64.b64decode(current_image)
            img       = Image.open(io.BytesIO(img_bytes))
            st.image(img, width=300)
        st.write(prompt)

    if st.session_state.chat_id:
        display_content = f"[Image attached]\n{prompt}" if current_image else prompt
        api_save_message(st.session_state.chat_id, "user", display_content)

    # Clear image after sending
    st.session_state.uploaded_image = None
    st.session_state.image_mime     = None

    # Stop button state
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    st.session_state.is_generating   = True
    st.session_state.stop_generation = False

    try:
        t_start  = time.time()
        auto_web = is_current_affairs(prompt)
        do_web   = use_web or auto_web

        if auto_web and not use_web:
            st.toast("🌐 Web search auto-triggered!", icon="🔍")

        history = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
        )

        # Build final prompt
        if current_image:
            image_prompt = f"""You are Talkio AI — an accurate, professional AI assistant.

Conversation History:
{history}

The user has attached an image. Analyze it carefully and answer the question.

User Question: {prompt}

Instructions:
- Describe what you see in the image if relevant.
- Answer the user's question about the image accurately.
- Be concise and professional.
"""
            source = "image"

        elif rag.is_loaded:
            retrieval    = rag.retrieve(prompt, k=7, use_web=do_web)
            context      = retrieval["context"]
            source       = retrieval["source"]
            final_prompt = f"""You are Talkio AI — a highly accurate, professional AI assistant.

Conversation History:
{history}

Retrieved Knowledge:
{context}

User Question: {prompt}

Instructions:
- Use retrieved knowledge first. Mention PDF source if relevant.
- If partially relevant, combine with general knowledge.
- If unrelated to PDFs, use general knowledge and mention it.
- Be concise, accurate and professional.
- Use bullet points for lists, **bold** for key terms.
"""
        elif do_web:
            retrieval    = rag.web_only_search(prompt)
            context      = retrieval["context"]
            source       = "web"
            final_prompt = f"""You are Talkio AI — a highly accurate AI assistant.

Conversation History:
{history}

Web Search Results:
{context}

User Question: {prompt}

Answer based on web results. Cite sources. Be concise and accurate.
"""
        else:
            source       = "direct"
            final_prompt = f"""You are Talkio AI — a helpful, accurate assistant.

Conversation History:
{history}

User Question: {prompt}

Answer accurately and concisely.
"""

        # -----------------------------
        # STREAM RESPONSE + STOP BUTTON
        # -----------------------------
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            stop_col1, stop_col2 = st.columns([6, 1])
            with stop_col2:
                stop_pressed = st.button("⏹ Stop", key=f"stop_{time.time()}")

            if stop_pressed:
                st.session_state.stop_generation = True

            answer      = ""
            full_answer = ""

            if current_image:
                # Image — no streaming (single call)
                with st.spinner("Analyzing image..."):
                    full_answer = ask_gemini_with_image(
                        image_prompt, current_image, current_image_mime
                    )
                response_placeholder.write(full_answer)
                answer = full_answer

            else:
                # Stream text response
                try:
                    stream = client.models.generate_content_stream(
                        model=MODEL,
                        contents=final_prompt,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            max_output_tokens=2048,
                        )
                    )

                    for chunk in stream:
                        # Check stop button
                        if st.session_state.stop_generation:
                            answer += "\n\n_[Generation stopped]_"
                            response_placeholder.markdown(answer)
                            st.toast("⏹ Generation stopped", icon="🛑")
                            break

                        if chunk.text:
                            answer += chunk.text
                            response_placeholder.markdown(answer)

                    full_answer = answer

                except Exception as e:
                    # Fallback to non-streaming
                    full_answer = ask_gemini(final_prompt)
                    response_placeholder.write(full_answer)
                    answer = full_answer

        latency = int((time.time() - t_start) * 1000)

        if rag.is_loaded and source not in ["cache", "web", "hybrid+web", "image"]:
            rag.cache_answer(prompt, full_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })

        if st.session_state.chat_id:
            api_save_message(st.session_state.chat_id, "assistant", full_answer)

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        st.session_state.is_generating   = False
        st.session_state.stop_generation = False