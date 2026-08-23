from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from typing import Mapping, Protocol


AUTH_DISABLED = "disabled"
AUTH_SHARED_TOKEN = "shared-token"
AUTH_OIDC_JWT = "oidc-jwt"
SUPPORTED_AUTH_MODES = frozenset({AUTH_DISABLED, AUTH_SHARED_TOKEN, AUTH_OIDC_JWT})


def _csv_env(name: str, default: tuple[str, ...] = ()) -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        return frozenset(default)
    return frozenset(value.strip() for value in raw.split(",") if value.strip())


def _claim_value(claims: Mapping[str, object], path: str) -> object | None:
    current: object = claims
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _roles_from_claim(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(role.strip() for role in value.split(",") if role.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(role).strip() for role in value if str(role).strip())
    return frozenset()


@dataclass(frozen=True)
class AuthIdentity:
    subject: str
    roles: frozenset[str]
    authenticated: bool
    auth_mode: str


@dataclass(frozen=True)
class AuthConfig:
    """Authentication/authorization configuration for the public application API.

    Authentication is deliberately disabled by default so adding this foundation
    does not break the existing clinical deployment before an identity provider is
    connected. Server-side case storage still refuses anonymous access.
    """

    mode: str = AUTH_DISABLED
    required_roles: frozenset[str] = frozenset({"clinician", "admin"})
    shared_token: str | None = None
    shared_subject: str = "service-user"
    shared_roles: frozenset[str] = frozenset({"clinician"})
    issuer: str | None = None
    audience: str | None = None
    jwks_url: str | None = None
    roles_claim: str = "roles"
    jwt_algorithms: tuple[str, ...] = ("RS256", "ES256")

    @classmethod
    def from_env(cls) -> "AuthConfig":
        mode = os.getenv("MEDDX_AUTH_MODE", AUTH_DISABLED).strip().lower()
        if mode not in SUPPORTED_AUTH_MODES:
            raise ValueError(
                "MEDDX_AUTH_MODE must be one of: " + ", ".join(sorted(SUPPORTED_AUTH_MODES))
            )

        algorithms = tuple(
            value.strip()
            for value in os.getenv("MEDDX_AUTH_JWT_ALGORITHMS", "RS256,ES256").split(",")
            if value.strip()
        )
        config = cls(
            mode=mode,
            required_roles=_csv_env("MEDDX_AUTH_REQUIRED_ROLES", ("clinician", "admin")),
            shared_token=os.getenv("MEDDX_AUTH_SHARED_TOKEN") or None,
            shared_subject=os.getenv("MEDDX_AUTH_SHARED_SUBJECT", "service-user").strip()
            or "service-user",
            shared_roles=_csv_env("MEDDX_AUTH_SHARED_ROLES", ("clinician",)),
            issuer=os.getenv("MEDDX_AUTH_ISSUER") or None,
            audience=os.getenv("MEDDX_AUTH_AUDIENCE") or None,
            jwks_url=os.getenv("MEDDX_AUTH_JWKS_URL") or None,
            roles_claim=os.getenv("MEDDX_AUTH_ROLES_CLAIM", "roles").strip() or "roles",
            jwt_algorithms=algorithms or ("RS256",),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode == AUTH_SHARED_TOKEN:
            if not self.shared_token:
                raise ValueError("MEDDX_AUTH_SHARED_TOKEN is required for shared-token auth")
            if not self.shared_subject:
                raise ValueError("MEDDX_AUTH_SHARED_SUBJECT must not be empty")
        elif self.mode == AUTH_OIDC_JWT:
            missing = [
                name
                for name, value in (
                    ("MEDDX_AUTH_ISSUER", self.issuer),
                    ("MEDDX_AUTH_AUDIENCE", self.audience),
                    ("MEDDX_AUTH_JWKS_URL", self.jwks_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "OIDC JWT auth is missing required environment: " + ", ".join(missing)
                )
            if not self.jwt_algorithms:
                raise ValueError("At least one JWT algorithm must be configured")


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class AuthProvider(Protocol):
    config: AuthConfig

    def authenticate(self, headers: Mapping[str, str]) -> AuthIdentity:
        ...


class DisabledAuthProvider:
    def __init__(self, config: AuthConfig):
        self.config = config

    def authenticate(self, headers: Mapping[str, str]) -> AuthIdentity:
        return AuthIdentity(
            subject="anonymous",
            roles=frozenset({"clinician"}),
            authenticated=False,
            auth_mode=AUTH_DISABLED,
        )


class SharedTokenAuthProvider:
    def __init__(self, config: AuthConfig):
        self.config = config
        if not config.shared_token:
            raise ValueError("Shared token auth requires a configured token")

    def authenticate(self, headers: Mapping[str, str]) -> AuthIdentity:
        token = bearer_token(headers.get("Authorization"))
        if not token or not hmac.compare_digest(token, self.config.shared_token or ""):
            raise AuthenticationError("Missing or invalid bearer token")
        return AuthIdentity(
            subject=self.config.shared_subject,
            roles=self.config.shared_roles,
            authenticated=True,
            auth_mode=AUTH_SHARED_TOKEN,
        )


class OIDCJWTAuthProvider:
    """Generic JWKS-backed JWT verifier for a future OIDC/Supabase connection."""

    def __init__(self, config: AuthConfig):
        self.config = config
        if not config.jwks_url:
            raise ValueError("OIDC JWT auth requires a JWKS URL")
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - dependency is part of the package
            raise RuntimeError("PyJWT is required for OIDC JWT authentication") from exc
        self._jwt = jwt
        self._jwk_client = jwt.PyJWKClient(config.jwks_url)

    def authenticate(self, headers: Mapping[str, str]) -> AuthIdentity:
        token = bearer_token(headers.get("Authorization"))
        if not token:
            raise AuthenticationError("Missing bearer token")

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = self._jwt.decode(
                token,
                signing_key.key,
                algorithms=list(self.config.jwt_algorithms),
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"require": ["exp", "iat", "sub"]},
            )
        except Exception as exc:
            raise AuthenticationError("Bearer token could not be verified") from exc

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise AuthenticationError("Bearer token is missing a valid subject")

        roles = _roles_from_claim(_claim_value(claims, self.config.roles_claim))
        return AuthIdentity(
            subject=subject.strip(),
            roles=roles,
            authenticated=True,
            auth_mode=AUTH_OIDC_JWT,
        )


def bearer_token(authorization_header: str | None) -> str | None:
    if not isinstance(authorization_header, str):
        return None
    scheme, separator, token = authorization_header.strip().partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def authorize(identity: AuthIdentity, required_roles: frozenset[str]) -> None:
    if required_roles and not identity.roles.intersection(required_roles):
        raise AuthorizationError("Authenticated identity is not authorized for this resource")


def build_auth_provider(config: AuthConfig | None = None) -> AuthProvider:
    config = config or AuthConfig.from_env()
    if config.mode == AUTH_DISABLED:
        return DisabledAuthProvider(config)
    if config.mode == AUTH_SHARED_TOKEN:
        return SharedTokenAuthProvider(config)
    if config.mode == AUTH_OIDC_JWT:
        return OIDCJWTAuthProvider(config)
    raise ValueError(f"Unsupported auth mode: {config.mode}")
