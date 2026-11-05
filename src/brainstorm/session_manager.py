from typing import Optional

from brainstorm_llm import Message


class ChatSession:
    
    def new(cls) -> 'ChatSession':
        pass
    
    def open(cls) -> 'ChatSession':
        pass
    
    def __init__(self, id: Optional[str] = None) -> None:
        self.id = id
    
    def messages(self) -> list[Message]:
        pass