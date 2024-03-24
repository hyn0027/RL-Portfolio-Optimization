import os
import importlib
from typing import Callable, Type, Dict, TypeVar


registered_networks: Dict[str, Type] = {}

T = TypeVar("T")


def register_network(network_name: str) -> Callable[[T], T]:
    def decorator(network_cls: T) -> T:
        registered_networks[network_name] = network_cls
        return network_cls

    return decorator


network_files = os.listdir(os.path.dirname(__file__))
network_files = [
    f[:-3] for f in network_files if f.endswith(".py") and f != "__init__.py"
]

for network_file in network_files:
    module = importlib.import_module(f"networks.{network_file}", package=__name__)
