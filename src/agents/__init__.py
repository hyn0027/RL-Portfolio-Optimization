import os
import importlib
from typing import Callable, Type, Dict


registered_agents: Dict[str, Type] = {}


def register_agent(agent_name: str) -> Callable[[Type], Type]:
    def decorator(agent_cls) -> Type:
        registered_agents[agent_name] = agent_cls
        return agent_cls

    return decorator


agent_files = os.listdir(os.path.dirname(__file__))
agent_files = [f[:-3] for f in agent_files if f.endswith(".py") and f != "__init__.py"]

for agent_file in agent_files:
    module = importlib.import_module(f"agents.{agent_file}", package=__name__)
