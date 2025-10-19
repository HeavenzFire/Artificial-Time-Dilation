"""
Web Interface

Web-based interface for the Spirit Angelus Framework, providing a no-code/low-code
spiritual guidance system accessible through modern web browsers.
"""

from .app import SpiritAngelusApp, create_app
from .routes import main_routes, api_routes, meditation_routes
from .templates import render_template, get_template
from .static import serve_static, get_static_path
from .auth import authenticate_user, authorize_user, get_user_session
from .websocket import WebSocketHandler, MeditationWebSocket

__all__ = [
    "SpiritAngelusApp",
    "create_app",
    "main_routes",
    "api_routes", 
    "meditation_routes",
    "render_template",
    "get_template",
    "serve_static",
    "get_static_path",
    "authenticate_user",
    "authorize_user",
    "get_user_session",
    "WebSocketHandler",
    "MeditationWebSocket",
]