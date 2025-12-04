from .simulator import SimulatorBase
from . import make_env as _make_env_module
from .make_env import MujocoEnv, list_available_robots, list_available_tasks


class _MakeEnvProxy:
    def __init__(self, module):
        super().__setattr__("_module", module)
        super().__setattr__("make_env", module.make_env)

    def __call__(self, *args, **kwargs):
        return self.make_env(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._module, item)

    def __setattr__(self, name, value):
        if name in {"_module", "make_env"}:
            super().__setattr__(name, value)
        else:
            setattr(self._module, name, value)


make_env = _MakeEnvProxy(_make_env_module)

__all__ = [
    'SimulatorBase',
    'make_env',
    'MujocoEnv',
    'list_available_robots',
    'list_available_tasks'
]
