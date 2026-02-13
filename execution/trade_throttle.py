class TradeThrottle:
    def __init__(self, min_interval_seconds=1.0):
        import time
        self.time = time
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_sent_at = {}
    def allow(self, key):
        now = self.time.time()
        last = self._last_sent_at.get(key)
        if last is None or (now - last) >= self.min_interval_seconds:
            self._last_sent_at[key] = now
            return True
        return False
