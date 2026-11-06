from typing import Any
from brainstorm_llm.integrations.langchain import ChatBrainStorm
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from pprint import pprint

def chat() -> None:
    model = ChatBrainStorm(
        model='deepseek/deepseek-v4.1-flash',
        provider='commandcode'
    )
    
    messages: list[BaseMessage] = [
        SystemMessage('You are concise.')
    ]
    
    while True:
        query = input("Ask agent: ").strip()
        
        if query == '/exit':
            break
        
        messages.append(HumanMessage(query))
        response: AIMessage = model.invoke(messages)
        messages.append(response)
        
        content: dict[Any, Any]
        
        for content in response.content:    # type: ignore
            if content['type'] == 'reasoning':
                print('<think>')
                print(content['reasoning'])
                print('</think>')
            
            if content['type'] == 'text':
                print(content['text'])
            

if __name__ == '__main__':
    chat()
