import os, urllib.parse, urllib.request, json
from financial_organism.utils.logger import get_logger
logger = get_logger("TELEGRAM")


def alert(msg):
    token = os.getenv("TELEGRAM_TOKEN","")
    chat_id = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        logger.info(f"Telegram disabled; alert={msg}")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = urllib.parse.urlencode({"chat_id": chat_id, "text": msg})
    with urllib.request.urlopen(f"{url}?{params}", timeout=5) as r:
        return r.status == 200


def poll_commands():
    """Check Telegram bot updates for commands and act on them.

    Currently we look for a /HALT_ALL message and trigger the
    HumanAIGovernance.halt() if seen.  This is a very small server-side
    poll; it is safe to call once per main loop iteration.
    """
    token = os.getenv("TELEGRAM_TOKEN", "")
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.load(r)
    except Exception as e:
        logger.debug(f"Failed to poll telegram updates: {e}")
        return []
    results = data.get("result", [])
    for item in results:
        msg = item.get("message", {})
        text = msg.get("text", "").strip()
        if text.upper() == "/HALT_ALL":
            logger.warning("Received /HALT_ALL command via Telegram")
            try:
                from financial_organism.governance.human_ai_governance import HumanAIGovernance
                HumanAIGovernance.halt("telegram")
            except Exception:
                pass
    return results
