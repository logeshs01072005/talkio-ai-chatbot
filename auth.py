import streamlit as st
import hashlib
import requests
from database import conn, cursor

API_URL = "http://127.0.0.1:8000"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def api_register_user(username, email=""):
    try:
        response = requests.post(
            f"{API_URL}/users",
            json={"username": username, "password": "hashed", "email": email}
        )
        if response.status_code == 201:
            return response.json()
        if response.status_code == 400:
            return api_get_user(username)
        return None
    except Exception as e:
        print(f"API Register Error: {e}")
        return None


def api_get_user(username):
    try:
        response = requests.get(f"{API_URL}/users")
        if response.status_code == 200:
            for user in response.json():
                if user["username"] == username:
                    return user
        return None
    except Exception as e:
        print(f"API Get User Error: {e}")
        return None


def api_create_chat(user_id, title="New Chat"):
    try:
        response = requests.post(
            f"{API_URL}/chats",
            json={"user_id": user_id, "title": title}
        )
        return response.json() if response.status_code == 201 else None
    except Exception as e:
        print(f"API Create Chat Error: {e}")
        return None


def login_page():

    # Session Variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None

    # Sidebar
    with st.sidebar:
        st.title("🤖 Talkio AI")
        st.markdown("---")
        st.markdown("**NAVIGATION**")
        menu = st.selectbox(
            "Select Page",
            ["Login", "Register", "Forgot Password"]
        )
        st.markdown("---")
        st.markdown("🔐 Secure login")
        st.markdown("🤖 AI-powered chat")
        st.markdown("📄 PDF analysis")
        st.markdown("🌐 Web search")

    # ==========================
    # REGISTER
    # ==========================
    if menu == "Register":

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("📝 Register")
            st.caption("Create your Talkio AI account")
            st.markdown("---")

            username = st.text_input("Username", placeholder="Enter username")
            email    = st.text_input("Email (optional)", placeholder="Enter email")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Create Account", use_container_width=True):
                if username == "" or password == "":
                    st.error("Please fill all required fields")
                else:
                    try:
                        cursor.execute(
                            "INSERT INTO users (username, password) VALUES (?, ?)",
                            (username, hash_password(password))
                        )
                        conn.commit()
                        api_user = api_register_user(username, email)
                        st.success(f"✅ Account created! Welcome {username}. Please login.")
                    except:
                        st.error("❌ Username already exists")

    # ==========================
    # LOGIN
    # ==========================
    elif menu == "Login":

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 Login")
            st.caption("Sign in to Talkio AI")
            st.markdown("---")

            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Sign In", use_container_width=True):
                cursor.execute(
                    "SELECT * FROM users WHERE username=? AND password=?",
                    (username, hash_password(password))
                )
                user = cursor.fetchone()

                if user:
                    st.session_state.logged_in = True
                    st.session_state.username  = username

                    api_user = api_get_user(username)
                    if not api_user:
                        api_user = api_register_user(username)

                    if api_user:
                        st.session_state.user_id = api_user["id"]
                        st.session_state.email   = api_user.get("email", "")
                        chat = api_create_chat(
                            user_id=api_user["id"],
                            title=f"{username}'s Chat"
                        )
                        if chat:
                            st.session_state.chat_id = chat["id"]

                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

    # ==========================
    # FORGOT PASSWORD
    # ==========================
    elif menu == "Forgot Password":

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔑 Reset Password")
            st.caption("Enter your details to reset password")
            st.markdown("---")

            username     = st.text_input("Username", placeholder="Enter username")
            new_password = st.text_input("New Password", type="password", placeholder="Enter new password")

            if st.button("Reset Password", use_container_width=True):
                cursor.execute("SELECT * FROM users WHERE username=?", (username,))
                user = cursor.fetchone()

                if user:
                    cursor.execute(
                        "UPDATE users SET password=? WHERE username=?",
                        (hash_password(new_password), username)
                    )
                    conn.commit()

                    api_user = api_get_user(username)
                    if api_user:
                        requests.put(
                            f"{API_URL}/users/{api_user['id']}",
                            json={"password": hash_password(new_password)}
                        )
                    st.success("✅ Password reset! Please login.")
                else:
                    st.error("❌ Username not found")