
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Abstract base class for action-selection policies. 
    """

    @abstractmethod
    def select_action(self) -> int:
        """
        action-selection method. Returns the arm to be selected
        """
        pass

    def update(self, arm, reward) -> None:
        """
        Update q-values.
        """
        pass
