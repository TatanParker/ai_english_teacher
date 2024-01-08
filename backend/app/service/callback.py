from queue import Queue

from langchain.callbacks.base import BaseCallbackHandler


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, queue: Queue) -> None:
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        """
        Handle new tokens from LLM.

        Args:
            token (str): The new token from LLM.
            **kwargs: Additional keyword arguments.
        """
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> bool:
        """
        Handle the end of LLM streaming.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self.queue.empty()
