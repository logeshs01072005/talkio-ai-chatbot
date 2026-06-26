from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid
import sqlite3

app = FastAPI(
    title="Talkio Chatbot API",
    description="FastAPI backend for Talkio AI Chatbot",
    version="1.0.0"
)

# -----------------------------
# CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# SQLite Database Setup
# -----------------------------
DB_PATH = "./talkio_api.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         TEXT PRIMARY KEY,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            email      TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Chats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         TEXT PRIMARY KEY,
            chat_id    TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)

    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# -----------------------------
# Pydantic Models
# -----------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str]
    created_at: str

class ChatCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class ChatUpdate(BaseModel):
    title: str

class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str

class MessageCreate(BaseModel):
    chat_id: str
    role: str
    content: str

class MessageUpdate(BaseModel):
    content: str

class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    created_at: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "🤖 Welcome to Talkio Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ============================================================
# USER APIs
# ============================================================

@app.get("/users", response_model=List[UserResponse], tags=["Users"])
def get_all_users():
    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return [dict(u) for u in users]


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: str):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id=?", (user_id,)
    ).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(user)


@app.post("/users", response_model=UserResponse, status_code=201, tags=["Users"])
def create_user(user: UserCreate):
    conn = get_db()

    # Check duplicate
    existing = conn.execute(
        "SELECT id FROM users WHERE username=?", (user.username,)
    ).fetchone()

    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    user_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO users (id, username, password, email, created_at) VALUES (?,?,?,?,?)",
        (user_id, user.username, user.password, user.email, created_at)
    )
    conn.commit()
    conn.close()

    return {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "created_at": created_at
    }


@app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
def update_user(user_id: str, user: UserUpdate):
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM users WHERE id=?", (user_id,)
    ).fetchone()

    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    if user.email is not None:
        conn.execute(
            "UPDATE users SET email=? WHERE id=?", (user.email, user_id)
        )
    if user.password is not None:
        conn.execute(
            "UPDATE users SET password=? WHERE id=?", (user.password, user_id)
        )

    conn.commit()
    updated = conn.execute(
        "SELECT * FROM users WHERE id=?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(updated)


@app.delete("/users/{user_id}", tags=["Users"])
def delete_user(user_id: str):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM users WHERE id=?", (user_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"message": f"User {user_id} deleted successfully"}


# ============================================================
# CHAT APIs
# ============================================================

@app.get("/chats", response_model=List[ChatResponse], tags=["Chats"])
def get_all_chats():
    conn = get_db()
    chats = conn.execute("SELECT * FROM chats").fetchall()
    conn.close()
    return [dict(c) for c in chats]


@app.get("/chats/user/{user_id}", response_model=List[ChatResponse], tags=["Chats"])
def get_chats_by_user(user_id: str):
    conn = get_db()
    chats = conn.execute(
        "SELECT * FROM chats WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(c) for c in chats]


@app.get("/chats/{chat_id}", response_model=ChatResponse, tags=["Chats"])
def get_chat(chat_id: str):
    conn = get_db()
    chat = conn.execute(
        "SELECT * FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    conn.close()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return dict(chat)


@app.post("/chats", response_model=ChatResponse, status_code=201, tags=["Chats"])
def create_chat(chat: ChatCreate):
    conn = get_db()

    user = conn.execute(
        "SELECT id FROM users WHERE id=?", (chat.user_id,)
    ).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    chat_id    = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO chats (id, user_id, title, created_at) VALUES (?,?,?,?)",
        (chat_id, chat.user_id, chat.title, created_at)
    )
    conn.commit()
    conn.close()

    return {
        "id": chat_id,
        "user_id": chat.user_id,
        "title": chat.title,
        "created_at": created_at
    }


@app.put("/chats/{chat_id}", response_model=ChatResponse, tags=["Chats"])
def update_chat(chat_id: str, chat: ChatUpdate):
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    conn.execute(
        "UPDATE chats SET title=? WHERE id=?", (chat.title, chat_id)
    )
    conn.commit()
    updated = conn.execute(
        "SELECT * FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    conn.close()
    return dict(updated)


@app.delete("/chats/{chat_id}", tags=["Chats"])
def delete_chat(chat_id: str):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()
    return {"message": f"Chat {chat_id} and its messages deleted"}


# ============================================================
# MESSAGE APIs
# ============================================================

@app.get("/messages/chat/{chat_id}", response_model=List[MessageResponse], tags=["Messages"])
def get_messages(chat_id: str):
    conn = get_db()
    chat = conn.execute(
        "SELECT id FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC",
        (chat_id,)
    ).fetchall()
    conn.close()
    return [dict(m) for m in messages]


@app.get("/messages/{message_id}", response_model=MessageResponse, tags=["Messages"])
def get_message(message_id: str):
    conn = get_db()
    msg = conn.execute(
        "SELECT * FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    conn.close()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    return dict(msg)


@app.post("/messages", response_model=MessageResponse, status_code=201, tags=["Messages"])
def create_message(message: MessageCreate):
    conn = get_db()
    chat = conn.execute(
        "SELECT id FROM chats WHERE id=?", (message.chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    if message.role not in ["user", "assistant"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")

    msg_id     = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (msg_id, message.chat_id, message.role, message.content, created_at)
    )
    conn.commit()
    conn.close()

    return {
        "id": msg_id,
        "chat_id": message.chat_id,
        "role": message.role,
        "content": message.content,
        "created_at": created_at
    }


@app.put("/messages/{message_id}", response_model=MessageResponse, tags=["Messages"])
def update_message(message_id: str, message: MessageUpdate):
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Message not found")
    conn.execute(
        "UPDATE messages SET content=? WHERE id=?", (message.content, message_id)
    )
    conn.commit()
    updated = conn.execute(
        "SELECT * FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    conn.close()
    return dict(updated)


@app.delete("/messages/{message_id}", tags=["Messages"])
def delete_message(message_id: str):
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM messages WHERE id=?", (message_id,)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Message not found")
    conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
    conn.commit()
    conn.close()
    return {"message": f"Message {message_id} deleted"}


@app.delete("/messages/chat/{chat_id}", tags=["Messages"])
def delete_all_messages(chat_id: str):
    conn = get_db()
    chat = conn.execute(
        "SELECT id FROM chats WHERE id=?", (chat_id,)
    ).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(status_code=404, detail="Chat not found")
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()
    return {"message": f"All messages in chat {chat_id} deleted"}