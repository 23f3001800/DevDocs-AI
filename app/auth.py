"""
JWT-based authentication with RBAC (user / admin roles).

Flow:
  1. POST /auth/register → create user → return JWT
  2. POST /auth/login    → verify password → return JWT
  3. POST /auth/logout   → revoke the presented token's jti
  4. Protected routes use Depends(require_role("user")) or require_role("admin")
  5. JWT carries: sub (username), role, jti (token id), exp, iat

WHY JWT instead of server-side sessions? Statelessness — any replica can
validate any token with no shared session store.

⚠️  The trade-off, stated plainly: a stateless JWT cannot be invalidated. This
module therefore keeps a small `revoked_tokens` denylist (see database.py) so
logout is real, at the cost of one indexed SELECT per authenticated request.

⚠️  The frontend stores the token in localStorage, which any XSS on the page can
read. That is the standard SPA trade-off, not a solved problem: keep
JWT_EXPIRY_HOURS short, and move to an httpOnly+SameSite cookie (plus CSRF
protection) if you ever handle data worth stealing.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import INSECURE_JWT_DEFAULT, get_settings
from app.database import create_user, get_user, is_token_revoked, revoke_token

log = logging.getLogger(__name__)

_settings = get_settings()

# Re-exported for tests and callers that sign/verify directly.
JWT_SECRET = _settings.jwt_secret
JWT_ALGORITHM = _settings.jwt_algorithm
JWT_EXPIRY_HOURS = _settings.jwt_expiry_hours
_INSECURE_JWT_DEFAULT = INSECURE_JWT_DEFAULT

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
    # bcrypt salts every hash (identical passwords → different hashes, defeating
    # rainbow tables) and is *deliberately slow*, which caps an offline attacker
    # at a few guesses/second/core. Never use SHA-256 here — it is fast, which
    # is exactly wrong for passwords.
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT helpers ──────────────────────────────────────────────
def create_token(username: str, role: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "role": role,
        # jti is what makes revocation possible — without a per-token id there
        # is nothing to put on the denylist.
        "jti": uuid.uuid4().hex,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Verify signature + expiry. NOTE: the payload is signed, not encrypted."""
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

    if is_token_revoked(payload.get("jti", "")):
        raise HTTPException(401, "Token has been revoked")

    username = payload.get("sub")
    if not username:
        raise HTTPException(401, "Invalid token — missing subject")

    user = get_user(username)
    if not user:
        raise HTTPException(401, "User not found")
    # Carry the claims through so /auth/logout can revoke this exact token.
    return {**user, "jti": payload.get("jti", ""), "exp": payload.get("exp")}


def require_role(minimum_role: str):
    """Dependency factory: require at least the given role.

    Role hierarchy: admin > user
    - require_role("user")  → allows user AND admin
    - require_role("admin") → allows admin ONLY
    """
    role_level = {"user": 1, "admin": 2}

    def dependency(user: dict = Depends(get_current_user)) -> dict:
        # .get(role, 0) means an unknown or corrupt role is denied — fail closed.
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
        # One message for both cases — revealing which half was wrong hands an
        # attacker a username oracle.
        raise HTTPException(401, "Invalid username or password")
    token = create_token(user["username"], user["role"])
    return AuthResponse(token=token, username=user["username"], role=user["role"])


def logout_user(user: dict) -> dict:
    """Revoke the presented token so it cannot be reused before expiry."""
    jti = user.get("jti")
    if not jti:
        raise HTTPException(400, "Token carries no jti and cannot be revoked")
    exp = user.get("exp")
    expires_at = (
        datetime.fromtimestamp(exp, UTC)
        if isinstance(exp, int | float)
        else datetime.now(UTC) + timedelta(hours=JWT_EXPIRY_HOURS)
    )
    revoke_token(jti, expires_at)
    log.info("Token revoked for user %s", user.get("username"))
    return {"status": "logged out"}
