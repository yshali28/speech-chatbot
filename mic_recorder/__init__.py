import base64
import os

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
_component_func = components.declare_component("mic_recorder", path=_FRONTEND_DIR)


def record_audio(key=None):
    """Renders the custom mic bar. Returns raw audio bytes once a
    recording is stopped, or None if nothing new has been recorded."""
    value = _component_func(key=key, default=None)
    if value is None:
        return None
    return base64.b64decode(value["audio_base64"])
