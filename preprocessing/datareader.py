from abc import ABC, abstractmethod


class AbstractDataReader(ABC):
    @abstractmethod
    def read_user_data(self, file:str):
        pass

    @abstractmethod
    def read_item_data(self, file:str):
        pass

    @abstractmethod
    def read_user_item_rating(self, file:str):
        pass
