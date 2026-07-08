import threading
import hashlib
import json

class APICache:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def _hash_query(self, query: str) -> str:
        # Normalize the query string: lowercase, strip padding, condense whitespace
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str):
        """Retrieve a cached response if it exists, otherwise return None."""
        key = self._hash_query(query)
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                # Return a deep copy to prevent mutation issues
                return json.loads(json.dumps(val))
            return None

    def set(self, query: str, response: dict):
        """Store a response in the cache."""
        key = self._hash_query(query)
        with self._lock:
            # Store a copy to freeze state
            self._cache[key] = json.loads(json.dumps(response))

    def clear(self):
        """Clear all entries in the cache."""
        with self._lock:
            self._cache.clear()
