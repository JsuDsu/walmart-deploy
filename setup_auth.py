"""Crea la tabla de usuarios y los usuarios por defecto en MySQL."""
from app.auth_db import init_auth_schema

if __name__ == "__main__":
    init_auth_schema()
    print("Tabla app_users lista. Usuarios: admin, analista, visor")
