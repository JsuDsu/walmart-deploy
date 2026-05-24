import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "cambia-esta-clave-en-produccion-walmart-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

ROLE_PERMISSIONS = {
    "admin": ["metrics", "predict", "retrain", "chat", "users", "reports"],
    "analyst": ["metrics", "predict", "chat", "reports"],
    "viewer": ["metrics", "chat", "reports"],
}

ROLE_LABELS = {
    "admin": "Administrador",
    "analyst": "Analista",
    "viewer": "Visor",
}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def permissions_for_role(role: str) -> list[str]:
    return ROLE_PERMISSIONS.get(role, [])


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
