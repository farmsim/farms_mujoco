"""Application"""

import glfw
from dm_control import viewer


class FarmsApplication(viewer.application.Application):
    """FARMS application"""

    def __init__(self, **kwargs):
        super().__init__(
            title=kwargs.pop('title', 'FARMS MuJoCo simulation'),
            width=kwargs.pop('width', 1000),
            height=kwargs.pop('height', 720),
        )

    def toggle_pause(self):
        """Toggle pause"""
        self._pause_subject.toggle()

    def close(self):
        """Close"""
        glfw.set_window_should_close(
            # pylint: disable=protected-access
            window=self._window._context.window,
            value=True,
        )
