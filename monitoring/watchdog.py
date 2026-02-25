"""Simple watchdog helper for restarting main processes.

This module is intentionally tiny; it can be invoked from a container or a
cron job.  The idea is to call ``watchdog.monitor(main_func)`` where
``main_func`` encapsulates the long‑running loop.  If an exception bubbles
up, the watchdog waits a few seconds and starts it again.

For zero‑budget deployments you can also use this script as a sanity
check before launching a container: the builtin auto‑restart behaviour of
many orchestrators (Docker, systemd, etc.) will already restart on failure,
so the watchdog simply wraps those semantics in Python.
"""
import time
import traceback


def monitor(func, *args, interval_seconds: float = 5.0, **kwargs):
    """Run ``func`` continuously; restart after any unhandled exception.

    Args:
        func: callable to run
        interval_seconds: pause between restart attempts
        *args/**kwargs: passed through to ``func``
    """
    while True:
        try:
            func(*args, **kwargs)
            # if func returns gracefully we also exit
            break
        except Exception:
            print("[WATCHDOG] process crashed, restarting in {interval_seconds}s")
            traceback.print_exc()
            time.sleep(interval_seconds)


if __name__ == '__main__':
    # Example usage: python -m financial_organism.monitoring.watchdog main
    import sys
    if len(sys.argv) > 1:
        modname = sys.argv[1]
        parts = modname.split('.')
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        if callable(mod):
            monitor(mod)
        else:
            print("Specified object is not callable.")
    else:
        print("Usage: python -m financial_organism.monitoring.watchdog <callable>")
