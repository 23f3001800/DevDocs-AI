"""
JWT-based authentication with RBAC (user / admin roles).

Flow:
  1. POST /auth/register → create user → return JWT
  2. POST /auth/login    → verify password → return JWT
  3. All protected routes use Depends(require_role("user")) or Depends(require_role("admin"))
  4. JWT contains: sub (username), role, exp (expiry)

WHY JWT instead of session cookies? Stateless auth — no server-side
session store needed. The frontend stores the token in localStorage
and sends it as Authorization: Bearer <token> on every request.
"""

import os
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.database import create_user, get_user

# ── Config ───────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

security = HTTPBearer(auto_error=False)


# ── Request / Response schemas ───────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str


# ── Password hashing ────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT helpers ──────────────────────────────────────────────
def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(UTC) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ── Auth dependency factories ────────────────────────────────
def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Extract and validate the JWT from the Authorization header."""
    if not creds:
        raise HTTPException(401, "Missing authorization header")
    payload = decode_token(creds.credentials)
    user = get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "User not found")
    return user


def require_role(minimum_role: str):
    """Dependency factory: require at least the given role.

    Role hierarchy: admin > user
    - require_role("user")  → allows user AND admin
    - require_role("admin") → allows admin ONLY
    """
    role_level = {"user": 1, "admin": 2}

    def dependency(user: dict = Depends(get_current_user)) -> dict:
        user_level = role_level.get(user["role"], 0)
        required_level = role_level.get(minimum_role, 0)
        if user_level < required_level:
            raise HTTPException(403, f"Requires '{minimum_role}' role — you have '{user['role']}'")
        return user

    return dependency


# ── Auth route handlers ──────────────────────────────────────
def register_user(req: RegisterRequest) -> AuthResponse:
    """Register a new user (default role: user)."""
    hashed = hash_password(req.password)
    try:
        user = create_user(req.username, hashed, role="user")
    except ValueError as e:
        raise HTTPException(409, str(e))
    token = create_token(user["username"], user["role"])
    return AuthResponse(token=token, username=user["username"], role=user["role"])


def login_user(req: LoginRequest) -> AuthResponse:
    """Authenticate and return a JWT."""
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid username or password")
    token = create_token(user["username"], user["role"])
    return AuthResponse(token=token, username=user["username"], role=user["role"])
