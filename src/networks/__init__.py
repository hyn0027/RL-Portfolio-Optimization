import os
import importlib
from typing import Callable, Type, Dict, TypeVar


registered_networks: Dict[str, Type] = {}

T = TypeVar("T")


def register_network(network_name: str) -> Callable[[T], T]:
    """the decorator to register a network

        to add a new class as a registered network,
        add the following code to the top of the file:

        .. code-block:: python

            @register_network("network_name")
            YourEnvClass(BaseEnv):
                ...

    Args:
        network_name (str): the name of the registered network

    Returns:
        Callable[[_T], _T]: the decorator to register the network
    """

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
