from abc import ABC, abstractmethod


class Address(ABC):
    """An Address is an identifier used to route messages to the correct subscriber.
    It must implement __hash__() method."""

    @abstractmethod
    def __hash__(self) -> int:
        pass
