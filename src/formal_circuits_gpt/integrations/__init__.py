"""External service integrations."""

from .github_client import GitHubClient
from .notification_service import NotificationService
from .auth_service import AuthService

__all__ = [
    "GitHubClient",
    "NotificationService", 
    "AuthService"
]