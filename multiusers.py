"""재정경제부 RAG 챗봇 - 멀티유저/멀티세션/저장 기능."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = LOG_DIR / log_name

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Constants & Prompts
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 10
BOT_NAME = "재정경제부 RAG 챗봇"

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _get_secret(key: str) -> str:
    """st.secrets 우선, 없으면 환경변수에서 읽습니다."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def _get_openai_key() -> str:
    return _get_secret("OPENAI_API_KEY")


def _get_supabase_creds() -> tuple[str, str]:
    return _get_secret("SUPABASE_URL"), _get_secret("SUPABASE_ANON_KEY")


# ---------------------------------------------------------------------------
# Password Hashing (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
_PBKDF2_ITERATIONS = 100_000


def _hash_password(password: str) -> str:
    """비밀번호를 salt$hash 형식으로 해시합니다."""
    salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"{salt}${key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """저장된 해시와 입력 비밀번호를 비교합니다."""
    try:
        if "$" not in stored_hash:
            logger.warning("저장된 해시에 구분자($)가 없음. 형식 오류: %r", stored_hash[:20])
            return False
        salt, key_hex = stored_hash.split("$", 1)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
        return key.hex() == key_hex
    except Exception as e:
        logger.warning("비밀번호 검증 예외: %s", e)
        return False


# ---------------------------------------------------------------------------
# Supabase Client
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase_client() -> Client | None:
    url, key = _get_supabase_creds()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.warning("Supabase 클라이언트 생성 실패: %s", e)
        return None


# ---------------------------------------------------------------------------
# User Management (DB-based, Supabase Auth 미사용)
# ---------------------------------------------------------------------------
def signup_user(login_id: str, password: str) -> tuple[bool, str]:
    """새 사용자를 등록합니다. (login_id, password_hash)"""
    sb = get_supabase_client()
    if not sb:
        return False, "Supabase 연결에 실패했습니다."

    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해주세요."
    if len(login_id) < 3:
        return False, "아이디는 3자 이상이어야 합니다."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상이어야 합니다."

    try:
        existing = (
            sb.table("users")
            .select("id")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            return False, "이미 사용 중인 아이디입니다."

        password_hash = _hash_password(password)
        res = (
            sb.table("users")
            .insert({"login_id": login_id, "password_hash": password_hash})
            .execute()
        )
        if res.data:
            return True, "회원가입이 완료되었습니다."
        return False, "회원가입에 실패했습니다."
    except Exception as e:
        logger.warning("회원가입 실패: %s", e)
        return False, f"오류: {e}"


def login_user(login_id: str, password: str) -> tuple[bool, str, str]:
    """로그인을 처리합니다. 성공 시 (True, user_id, login_id) 반환."""
    sb = get_supabase_client()
    if not sb:
        return False, "", "Supabase 연결에 실패했습니다."

    login_id = login_id.strip()
    if not login_id or not password:
        return False, "", "아이디와 비밀번호를 모두 입력해주세요."

    try:
        res = (
            sb.table("users")
            .select("id, password_hash")
            .eq("login_id", login_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            return False, "", f"등록된 아이디를 찾을 수 없습니다: '{login_id}'"

        row = res.data[0]
        stored_hash: str = row.get("password_hash", "")

        if not stored_hash:
            logger.warning("password_hash 컬럼이 비어 있음: login_id=%s", login_id)
            return False, "", "계정 데이터가 올바르지 않습니다. 관리자에게 문의하세요."

        if not _verify_password(password, stored_hash):
            logger.warning("비밀번호 불일치: login_id=%s, hash_len=%d", login_id, len(stored_hash))
            return False, "", "비밀번호가 올바르지 않습니다."

        return True, row["id"], login_id
    except Exception as e:
        logger.warning("로그인 실패: %s", e)
        return False, "", f"로그인 오류: {e}"


# ---------------------------------------------------------------------------
# Session Management (DB) — 모든 쿼리에 user_id 필터 적용
# ---------------------------------------------------------------------------
def load_sessions_from_db(user_id: str) -> list[dict]:
    """해당 사용자의 세션 목록을 Supabase에서 로드합니다."""
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = (
            sb.table("sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning("세션 로드 실패: %s", e)
        return []


def ensure_session_exists(session_id: str, user_id: str) -> bool:
    """sessions 테이블에 해당 session_id 행이 없으면 임시 제목으로 미리 생성합니다."""
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        sb.table("sessions").upsert(
            {
                "id": session_id,
                "user_id": user_id,
                "title": "임시 세션",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict="id",
        ).execute()
        return True
    except Exception as e:
        logger.warning("세션 사전 생성 실패: %s", e)
        return False


def save_session_to_db(
    session_id: str, user_id: str, title: str, messages: list[dict]
) -> bool:
    """현재 세션을 Supabase에 저장합니다."""
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        sb.table("sessions").upsert(
            {
                "id": session_id,
                "user_id": user_id,
                "title": title,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict="id",
        ).execute()

        if messages:
            msg_rows = [
                {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": m["role"],
                    "content": m["content"],
                    "order_index": i,
                    "created_at": datetime.utcnow().isoformat(),
                }
                for i, m in enumerate(messages)
            ]
            sb.table("chat_messages").insert(msg_rows).execute()

        return True
    except Exception as e:
        logger.warning("세션 저장 실패: %s", e)
        return False


def load_session_messages(session_id: str, user_id: str) -> list[dict]:
    """세션의 채팅 메시지를 user_id 조건으로 로드합니다."""
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = (
            sb.table("chat_messages")
            .select("role, content")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("order_index")
            .execute()
        )
        return [{"role": r["role"], "content": r["content"]} for r in (res.data or [])]
    except Exception as e:
        logger.warning("메시지 로드 실패: %s", e)
        return []


def load_session_files(session_id: str) -> list[str]:
    """세션에 업로드된 파일 목록을 로드합니다."""
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = (
            sb.table("vector_documents")
            .select("file_name")
            .eq("session_id", session_id)
            .execute()
        )
        seen: set[str] = set()
        files: list[str] = []
        for row in res.data or []:
            fn = row["file_name"]
            if fn not in seen:
                seen.add(fn)
                files.append(fn)
        return files
    except Exception as e:
        logger.warning("세션 파일 로드 실패: %s", e)
        return []


def delete_session_from_db(session_id: str, user_id: str) -> bool:
    """세션과 관련 데이터를 user_id 검증 후 삭제합니다."""
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        # 해당 세션이 현재 사용자 소유인지 확인
        check = (
            sb.table("sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not check.data:
            logger.warning("세션 삭제 권한 없음: session_id=%s, user_id=%s", session_id, user_id)
            return False

        sb.table("chat_messages").delete().eq("session_id", session_id).execute()
        sb.table("vector_documents").delete().eq("session_id", session_id).execute()
        sb.table("sessions").delete().eq("id", session_id).execute()
        return True
    except Exception as e:
        logger.warning("세션 삭제 실패: %s", e)
        return False


# ---------------------------------------------------------------------------
# Vector DB (Supabase)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embeddings_model() -> OpenAIEmbeddings | None:
    key = _get_openai_key()
    if not key:
        return None
    return OpenAIEmbeddings(model="text-embedding-ada-002", api_key=key)


def process_and_store_pdfs(
    uploaded_files: list[Any], session_id: str, user_id: str
) -> list[str]:
    """PDF 파일들을 처리하고 Supabase vector_documents에 저장합니다."""
    sb = get_supabase_client()
    if not sb:
        st.error("Supabase 연결이 없습니다. 환경 변수를 확인해주세요.")
        return []

    embeddings = get_embeddings_model()
    if not embeddings:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return []

    if not ensure_session_exists(session_id, user_id):
        st.error("세션 초기화에 실패했습니다. Supabase 연결을 확인해주세요.")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_files: list[str] = []

    for uf in uploaded_files:
        file_name = uf.name
        suffix = Path(file_name).suffix.lower() or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        except Exception as e:
            logger.warning("PDF 로드 실패 %s: %s", file_name, e)
            st.warning(f"'{file_name}' 로드 실패: {e}")
            continue
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not docs:
            continue

        splits = splitter.split_documents(docs)

        for doc in splits:
            doc.metadata["file_name"] = file_name
            doc.metadata["session_id"] = session_id

        for i in range(0, len(splits), BATCH_SIZE):
            batch = splits[i : i + BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = embeddings.embed_documents(texts)
            except Exception as e:
                logger.warning("임베딩 생성 실패: %s", e)
                continue

            rows = [
                {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": doc.page_content,
                    "embedding": emb,
                    "metadata": doc.metadata,
                    "created_at": datetime.utcnow().isoformat(),
                }
                for doc, emb in zip(batch, batch_embeddings)
            ]

            try:
                sb.table("vector_documents").insert(rows).execute()
            except Exception as e:
                logger.warning("벡터 저장 실패 (배치 %d): %s", i, e)

        processed_files.append(file_name)

    return processed_files


def search_vector_db(query: str, session_id: str, k: int = 10) -> list[str]:
    """Supabase RPC를 통해 세션별 벡터 검색을 수행합니다."""
    sb = get_supabase_client()
    embeddings = get_embeddings_model()
    if not sb or not embeddings:
        return []

    try:
        query_embedding = embeddings.embed_query(query)
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        if res.data:
            return [row["content"] for row in res.data]
    except Exception as e:
        logger.warning("RPC 벡터 검색 실패: %s → 일반 쿼리로 폴백", e)
        try:
            fallback = (
                sb.table("vector_documents")
                .select("content")
                .eq("session_id", session_id)
                .limit(k)
                .execute()
            )
            return [r["content"] for r in (fallback.data or [])]
        except Exception as e2:
            logger.warning("폴백 검색도 실패: %s", e2)

    return []


def get_vectordb_files(user_id: str) -> list[dict]:
    """현재 사용자의 벡터 DB 파일 목록을 반환합니다."""
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        # 사용자의 세션 ID 목록을 먼저 조회
        session_res = (
            sb.table("sessions")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        session_ids = [s["id"] for s in (session_res.data or [])]
        if not session_ids:
            return []

        res = sb.table("vector_documents").select("file_name, session_id").execute()
        seen: set[tuple] = set()
        files: list[dict] = []
        for row in res.data or []:
            if row["session_id"] not in session_ids:
                continue
            key = (row["file_name"], row["session_id"])
            if key not in seen:
                seen.add(key)
                files.append({"file_name": row["file_name"], "session_id": row["session_id"]})
        return files
    except Exception as e:
        logger.warning("vectordb 파일 목록 조회 실패: %s", e)
        return []


# ---------------------------------------------------------------------------
# LLM Functions
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.7) -> ChatOpenAI | None:
    key = _get_openai_key()
    if not key:
        return None
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=key)


def generate_session_title(first_q: str, first_a: str) -> str:
    """첫 번째 Q&A를 요약하여 세션 제목을 생성합니다."""
    llm = get_llm(temperature=0.3)
    if not llm:
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    prompt = (
        "다음 첫 번째 질문과 답변을 바탕으로 대화 세션의 제목을 한국어로 15자 이내로 만들어주세요.\n"
        "제목만 출력하세요. 다른 텍스트는 없어야 합니다.\n\n"
        f"[질문]\n{first_q[:500]}\n\n[답변]\n{first_a[:500]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = (getattr(out, "content", str(out)) or "").strip()
        return title[:50] if title else f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        logger.warning("세션 제목 생성 실패: %s", e)
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def generate_followup_questions(llm: Any, user_q: str, answer: str) -> str:
    """답변 후 후속 질문 3개를 생성합니다."""
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators((getattr(out, "content", str(out)) or "").strip())
        if raw:
            return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw}\n"
    except Exception as e:
        logger.warning("후속 질문 생성 실패: %s", e)
    return ""


def _format_memory_block(messages: list[dict], max_items: int = 20) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content[:500]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        # 인증 상태
        "authenticated": False,
        "current_user_id": None,
        "current_user_login_id": None,
        # 채팅 상태
        "chat_history": [],
        "conversation_memory": [],
        "current_session_id": str(uuid.uuid4()),
        "session_title": None,
        "processed_files": [],
        "sessions_list": [],
        "selected_session_id": None,
        "show_vectordb": False,
        "sessions_loaded": False,
        "rag_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Login / Signup Page
# ---------------------------------------------------------------------------
def _check_missing_keys() -> list[str]:
    missing = []
    if not _get_secret("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not _get_secret("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    if not _get_openai_key():
        missing.append("OPENAI_API_KEY")
    return missing


def render_login_page() -> None:
    """로그인/회원가입 페이지를 렌더링합니다."""
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
.login-container {
    max-width: 420px;
    margin: 0 auto;
    padding: 2rem 2.5rem;
    background: #1e2a3a;
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
div.stButton > button {
    background-color: #4a90d9;
    color: #ffffff;
    border-radius: 6px;
    font-weight: bold;
    border: none;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #357abd;
    color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # 헤더
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=120)
        else:
            st.markdown("### 🏛️")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0; padding-top:10px;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;"> RAG 챗봇</span>
</h1>
<p style="text-align:center; color:#888; margin:4px 0 20px 0; font-size:0.9rem;">
  Supabase 벡터 DB | OpenAI GPT-4o-mini | 멀티유저 세션
</p>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    # 누락된 환경 변수 경고
    missing = _check_missing_keys()
    if missing:
        st.warning(f"⚠️ 누락된 환경 변수: {', '.join(missing)}\n\n`.env` 파일 또는 Streamlit Secrets를 확인해주세요.")

    # 로그인 / 회원가입 탭
    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_signup = st.tabs(["🔐 로그인", "📝 회원가입"])

        with tab_login:
            st.markdown("")
            login_id = st.text_input("아이디", key="login_id_input", placeholder="아이디를 입력하세요")
            password = st.text_input(
                "비밀번호", type="password", key="login_pw_input", placeholder="비밀번호를 입력하세요"
            )
            st.markdown("")
            if st.button("로그인", key="btn_login", use_container_width=True):
                if not login_id or not password:
                    st.error("아이디와 비밀번호를 입력해주세요.")
                else:
                    with st.spinner("로그인 중..."):
                        ok, user_id, uid = login_user(login_id, password)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.current_user_id = user_id
                        st.session_state.current_user_login_id = uid
                        st.session_state.sessions_loaded = False
                        st.rerun()
                    else:
                        st.error(uid)  # uid에 오류 메시지가 담겨 옴

        with tab_signup:
            st.markdown("")
            new_id = st.text_input("아이디", key="signup_id_input", placeholder="3자 이상")
            new_pw = st.text_input(
                "비밀번호", type="password", key="signup_pw_input", placeholder="6자 이상"
            )
            new_pw2 = st.text_input(
                "비밀번호 확인", type="password", key="signup_pw2_input", placeholder="비밀번호 재입력"
            )
            st.markdown("")
            if st.button("회원가입", key="btn_signup", use_container_width=True):
                if new_pw != new_pw2:
                    st.error("비밀번호가 일치하지 않습니다.")
                else:
                    with st.spinner("회원가입 중..."):
                        ok, msg = signup_user(new_id, new_pw)
                    if ok:
                        st.success(f"✅ {msg} 로그인 탭에서 로그인해주세요.")
                    else:
                        st.error(msg)


# ---------------------------------------------------------------------------
# Main App (로그인 후)
# ---------------------------------------------------------------------------
def render_main_app() -> None:
    user_id: str = st.session_state.current_user_id
    user_login_id: str = st.session_state.current_user_login_id

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button {
    background-color: #4a90d9;
    color: #ffffff;
    border-radius: 6px;
    font-weight: bold;
    border: none;
}
div.stButton > button:hover {
    background-color: #357abd;
    color: #ffffff;
}
.session-info {
    background-color: #1e2a3a;
    border-radius: 8px;
    padding: 10px;
    font-size: 0.85rem;
    color: #aaa;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # ---- 헤더 ----
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=150)
        else:
            st.markdown("### 🏛️")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0; padding-top:10px;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;"> RAG 챗봇</span>
</h1>
<p style="text-align:center; color:#888; margin:4px 0 0 0; font-size:0.9rem;">
  Supabase 벡터 DB | OpenAI GPT-4o-mini | PDF RAG | 멀티유저 세션
</p>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    # 앱 시작 시 세션 목록 1회 로드
    if not st.session_state.sessions_loaded:
        st.session_state.sessions_list = load_sessions_from_db(user_id)
        st.session_state.sessions_loaded = True

    # ---- 사이드바 ----
    with st.sidebar:
        # 사용자 정보 & 로그아웃
        st.markdown(f"👤 **{user_login_id}** 님")
        if st.button("🚪 로그아웃", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("## 🗂️ 세션 관리")

        sessions: list[dict] = st.session_state.sessions_list
        session_options: dict[str, str] = {s["title"]: s["id"] for s in sessions}
        session_labels: list[str] = ["(선택 없음)"] + list(session_options.keys())

        selected_label = st.selectbox(
            "저장된 세션 선택",
            session_labels,
            key="session_selectbox",
        )
        if selected_label != "(선택 없음)":
            st.session_state.selected_session_id = session_options[selected_label]
        else:
            st.session_state.selected_session_id = None

        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            btn_save = st.button("💾 세션저장", use_container_width=True)
        with col2:
            btn_load = st.button("📂 세션로드", use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            btn_delete = st.button("🗑️ 세션삭제", use_container_width=True)
        with col4:
            btn_reset = st.button("🔄 화면초기화", use_container_width=True)

        btn_vectordb = st.button("🗄️ vectordb", use_container_width=True)

        st.markdown("---")
        st.markdown("## 📄 PDF 업로드")

        rag_col1, rag_col2 = st.columns(2)
        with rag_col1:
            if st.button(
                "✅ RAG 사용" if st.session_state.rag_enabled else "RAG 사용",
                use_container_width=True,
                type="primary" if st.session_state.rag_enabled else "secondary",
            ):
                st.session_state.rag_enabled = True
                st.rerun()
        with rag_col2:
            if st.button(
                "✅ 사용 안 함" if not st.session_state.rag_enabled else "사용 안 함",
                use_container_width=True,
                type="primary" if not st.session_state.rag_enabled else "secondary",
            ):
                st.session_state.rag_enabled = False
                st.rerun()

        uploaded_files = st.file_uploader(
            "PDF 파일 선택",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("📥 파일 처리하기", use_container_width=True):
            if not uploaded_files:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                with st.spinner(f"{len(uploaded_files)}개 파일 처리 중..."):
                    new_files = process_and_store_pdfs(
                        list(uploaded_files),
                        st.session_state.current_session_id,
                        user_id,
                    )
                if new_files:
                    for f in new_files:
                        if f not in st.session_state.processed_files:
                            st.session_state.processed_files.append(f)
                    st.success(f"{len(new_files)}개 파일 처리 완료!")
                else:
                    st.error("파일 처리에 실패했습니다.")

        if st.session_state.processed_files:
            st.markdown("**처리된 파일**")
            for f in st.session_state.processed_files:
                st.caption(f"• {f}")

        st.markdown("---")

        st.markdown("**현재 세션 정보**")
        st.caption(f"세션 ID: `{st.session_state.current_session_id[:8]}...`")
        st.caption(f"메시지 수: {len(st.session_state.chat_history)}")
        if st.session_state.session_title:
            st.caption(f"세션명: **{st.session_state.session_title}**")
        rag_status = "🟢 RAG 사용 중" if st.session_state.rag_enabled else "⚪ RAG 사용 안 함"
        st.caption(rag_status)

        openai_key = _get_openai_key()
        sb_url, sb_key = _get_supabase_creds()
        st.markdown("")
        if not openai_key:
            st.warning("⚠️ OPENAI_API_KEY 누락")
        if not sb_url or not sb_key:
            st.warning("⚠️ SUPABASE 설정 누락")
        if openai_key and sb_url and sb_key:
            st.success("✅ 환경 변수 정상")

    # ---- 버튼 처리 ----

    # 💾 세션저장
    if btn_save:
        if not st.session_state.chat_history:
            st.warning("저장할 대화 내용이 없습니다.")
        elif st.session_state.session_title:
            st.info(f"이미 저장된 세션입니다: **{st.session_state.session_title}**")
        else:
            messages = st.session_state.chat_history
            first_q = next((m["content"] for m in messages if m["role"] == "user"), "")
            first_a = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            with st.spinner("세션 제목 생성 중..."):
                title = generate_session_title(first_q, first_a)
            session_id = st.session_state.current_session_id
            with st.spinner("Supabase에 세션 저장 중..."):
                ok = save_session_to_db(session_id, user_id, title, messages)
            if ok:
                st.session_state.session_title = title
                st.session_state.sessions_list = load_sessions_from_db(user_id)
                st.success(f"✅ 세션 저장 완료: **{title}**")
                st.rerun()
            else:
                st.error("세션 저장에 실패했습니다. Supabase 연결을 확인해주세요.")

    # 📂 세션로드
    if btn_load:
        sel_id = st.session_state.selected_session_id
        if not sel_id:
            st.warning("로드할 세션을 선택하세요.")
        else:
            with st.spinner("세션 로드 중..."):
                msgs = load_session_messages(sel_id, user_id)
                session_files = load_session_files(sel_id)
            if msgs:
                st.session_state.chat_history = msgs
                st.session_state.conversation_memory = msgs.copy()
                st.session_state.current_session_id = sel_id
                st.session_state.processed_files = session_files
                for s in st.session_state.sessions_list:
                    if s["id"] == sel_id:
                        st.session_state.session_title = s["title"]
                        break
                st.success(f"✅ {len(msgs)}개 메시지 로드 완료!")
                st.rerun()
            else:
                st.warning("해당 세션에 저장된 메시지가 없습니다.")

    # 🗑️ 세션삭제
    if btn_delete:
        sel_id = st.session_state.selected_session_id
        if not sel_id:
            st.warning("삭제할 세션을 선택하세요.")
        else:
            with st.spinner("세션 삭제 중..."):
                ok = delete_session_from_db(sel_id, user_id)
            if ok:
                st.session_state.sessions_list = load_sessions_from_db(user_id)
                if st.session_state.current_session_id == sel_id:
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.session_state.session_title = None
                    st.session_state.processed_files = []
                st.session_state.selected_session_id = None
                st.success("✅ 세션이 삭제되었습니다.")
                st.rerun()
            else:
                st.error("세션 삭제에 실패했습니다.")

    # 🔄 화면초기화
    if btn_reset:
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.session_title = None
        st.session_state.processed_files = []
        st.session_state.selected_session_id = None
        st.session_state.show_vectordb = False
        st.rerun()

    # 🗄️ vectordb 토글
    if btn_vectordb:
        st.session_state.show_vectordb = not st.session_state.show_vectordb

    # ---- vectordb 파일 표시 ----
    if st.session_state.show_vectordb:
        with st.expander("🗄️ Vector DB 파일 목록", expanded=True):
            with st.spinner("조회 중..."):
                files = get_vectordb_files(user_id)
            if files:
                st.markdown(f"총 **{len(files)}**개 파일이 Vector DB에 저장되어 있습니다.")
                for f in files:
                    st.text(f"📄 {f['file_name']}  (세션: {f['session_id'][:8]}...)")
            else:
                st.info("Vector DB에 저장된 파일이 없습니다.")

    st.markdown("")

    # ---- 채팅 히스토리 표시 ----
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    # ---- 채팅 입력 ----
    user_input = st.chat_input("질문을 입력하세요...")
    if not user_input:
        return

    openai_key = _get_openai_key()
    if not openai_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일 또는 Streamlit Secrets를 확인해주세요.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    llm = get_llm()
    if not llm:
        with st.chat_message("assistant"):
            st.error("LLM 초기화에 실패했습니다. API 키를 확인해주세요.")
        return

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            context = ""
            if st.session_state.rag_enabled and st.session_state.processed_files:
                with st.spinner("📚 관련 문서 검색 중..."):
                    context_docs = search_vector_db(
                        user_input,
                        st.session_state.current_session_id,
                    )
                context = "\n\n".join(context_docs) if context_docs else ""

            mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])

            if context:
                sys_content = (
                    f"{ANSWER_STYLE_SYSTEM}\n\n"
                    "아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. "
                    "참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.\n\n"
                    f"[대화 맥락]\n{mem_txt or '(없음)'}\n\n"
                    f"[참고 문서]\n{context}"
                )
            else:
                sys_content = (
                    f"{ANSWER_STYLE_SYSTEM}\n\n"
                    f"[대화 맥락]\n{mem_txt or '(없음)'}"
                )

            msgs = [SystemMessage(content=sys_content), HumanMessage(content=user_input)]

            acc = ""
            for chunk in llm.stream(msgs):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    acc += piece
                    placeholder.markdown(remove_separators(acc) + "▌")

            full_answer = remove_separators(acc)
            placeholder.markdown(full_answer)

            if full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = generate_followup_questions(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as e:
            logger.warning("답변 생성 실패: %s", e)
            full_answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{e}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title=BOT_NAME,
        page_icon="🏛️",
        layout="wide",
    )
    _init_session_state()

    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_main_app()


if __name__ == "__main__":
    main()
