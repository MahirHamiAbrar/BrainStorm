from uuid import uuid4
from typing import Optional, Self

from brainstorm_llm import Message


class ChatSession:
    
    @classmethod
    def new(cls) -> 'ChatSession':
        _id = str(uuid4())
        cs = ChatSession(id=_id)
        return cs
    
    @classmethod
    def open(cls) -> 'ChatSession':
        pass
    
    def __init__(self, id: str) -> None:
        self.id = id
    
    def messages(self) -> list[Message]:
        pass