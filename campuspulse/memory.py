from collections import defaultdict, deque

class ConversationMemory:
    """Small in-process chat history for a demo; not persistent storage."""
    def __init__(self, max_messages=12):
        self._history = defaultdict(lambda: deque(maxlen=max_messages))

    def get(self, key):
        return list(self._history[key])

    def add(self, key, role, content):
        self._history[key].append({"role": role, "content": content})
