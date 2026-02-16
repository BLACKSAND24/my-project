import os, urllib.parse, urllib.request
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
