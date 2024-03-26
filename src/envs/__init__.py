import os
import importlib
from typing import Callable, Type, Dict, TypeVar

registered_envs: Dict[str, Type] = {}

T = TypeVar("T")


def register_env(environment_name: str) -> Callable[[T], T]:
    """the decorator to register an environment

        to add a new class as a registered environment,
        ensure the class is a derivative type of envs.BaseEnv,
        add the following code to the top of the file:

        .. code-block:: python

            @register_env("env_name")
            YourEnvClass(BaseEnv):
                ...

    Args:
        environment_name (str): the name of the registered environment

    Returns:
        Callable[[_T], _T]: the decorator to register the environment
    """

    def decorator(environment_cls: T) -> T:
        registered_envs[environment_name] = environment_cls
        return environment_cls

    return decorator


env_files = os.listdir(os.path.dirname(__file__))
env_files = [f[:-3] for f in env_files if f.endswith(".py") and f != "__init__.py"]

for env_file in env_files:
    module = importlib.import_module(f"envs.{env_file}", package=__name__)
