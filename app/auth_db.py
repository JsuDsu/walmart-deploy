import logging

import mysql.connector
from mysql.connector import Error
from .db_config import DB_CONFIG
from .security import ROLE_LABELS, ROLE_PERMISSIONS, hash_password

logger = logging.getLogger(__name__)

DEFAULT_USERS = [
    ("admin", "admin123", "admin"),
    ("analista", "analista123", "analyst"),
    ("visor", "visor123", "viewer"),
]


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_auth_schema() -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('admin', 'analyst', 'viewer') NOT NULL,
                is_active TINYINT(1) NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM app_users")
        count = cursor.fetchone()[0]
        if count == 0:
            insert_sql = """
                INSERT INTO app_users (username, password_hash, role)
                VALUES (%s, %s, %s)
            """
            for username, password, role in DEFAULT_USERS:
                cursor.execute(insert_sql, (username, hash_password(password), role))
            conn.commit()
            logger.info("Usuarios por defecto creados: admin, analista, visor")
    finally:
        if conn.is_connected():
            conn.close()


def get_user_by_username(username: str) -> dict | None:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, username, password_hash, role, is_active
            FROM app_users
            WHERE username = %s
            LIMIT 1
            """,
            (username,),
        )
        return cursor.fetchone()
    except Error as exc:
        logger.error("Error consultando usuario: %s", exc)
        return None
    finally:
        if conn.is_connected():
            conn.close()


def list_users() -> list[dict]:
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, username, role, is_active, created_at
            FROM app_users
            ORDER BY username
            """
        )
        rows = cursor.fetchall()
        for row in rows:
            row["role_label"] = ROLE_LABELS.get(row.get("role"), row.get("role"))
            if row.get("created_at"):
                row["created_at"] = row["created_at"].isoformat()
        return rows
    finally:
        if conn.is_connected():
            conn.close()


def create_user(username: str, password: str, role: str) -> dict:
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"Rol inválido: {role}")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO app_users (username, password_hash, role)
            VALUES (%s, %s, %s)
            """,
            (username, hash_password(password), role),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "username": username, "role": role}
    finally:
        if conn.is_connected():
            conn.close()
