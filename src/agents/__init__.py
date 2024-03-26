import os
import importlib
from typing import Callable, Type, Dict, TypeVar


registered_agents: Dict[str, Type] = {}

_T = TypeVar("_T")


def register_agent(agent_name: str) -> Callable[[_T], _T]:
    """the decorator to register an agent

        to add a new class as a registered agent,
        ensure the class is a derivative type of agents.BaseAgent,
        add the following code to the top of the file:

        .. code-block:: python

            @register_agent("agent_name")
            YourAgentClass(BaseAgent):
                ...

    Args:
        agent_name (str): the name of the registered agent

    Returns:
        Callable[[_T], _T]: the decorator to register the agent
    """

    def decorator(agent_cls: _T) -> _T:
        """register the agent

        Args:
            agent_cls (_T): the class of the agent
        Returns:
            _T: the class of the agent
        """
        registered_agents[agent_name] = agent_cls
        return agent_cls

    return decorator


agent_files = os.listdir(os.path.dirname(__file__))
agent_files = [f[:-3] for f in agent_files if f.endswith(".py") and f != "__init__.py"]

for _agent_file in agent_files:
    module = importlib.import_module(f"agents.{_agent_file}", package=__name__)
