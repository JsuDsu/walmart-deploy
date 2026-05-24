from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .auth_db import get_user_by_username
from .security import decode_access_token, permissions_for_role

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_username(payload["sub"])
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado o inactivo",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role = user["role"]
    return {
        "id": user["id"],
        "username": user["username"],
        "role": role,
        "permissions": permissions_for_role(role),
    }


def require_permission(permission: str):
    async def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        if permission not in current_user["permissions"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tienes permiso para realizar esta acción",
            )
        return current_user

    return _checker
