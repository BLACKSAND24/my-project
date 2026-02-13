from config import CONFIG
CHECKLIST = [
    {"id":"mode_is_live","description":"System is explicitly configured for LIVE mode","validator":lambda c: c.get("mode")=="LIVE"},
    {"id":"exchange_credentials_present","description":"Broker credentials are present","validator":lambda c: bool(c.get("exchange_api_key")) and bool(c.get("exchange_api_secret"))},
    {"id":"risk_acknowledged","description":"Risk acknowledgement explicitly enabled","validator":lambda c: bool(c.get("risk_acknowledged"))},
    {"id":"max_exposure_sane","description":"Max exposure <= 50%","validator":lambda c: float(c.get("max_total_exposure",1.0)) <= 0.5},
]
def build_context(overrides=None):
    c = {
        "mode": CONFIG.get("MODE","PAPER"),
        "exchange_api_key": CONFIG.get("EXCHANGE_API_KEY",""),
        "exchange_api_secret": CONFIG.get("EXCHANGE_API_SECRET",""),
        "risk_acknowledged": CONFIG.get("RISK_ACKNOWLEDGED",False),
        "max_total_exposure": CONFIG.get("MAX_TOTAL_EXPOSURE",1.0),
    }
    if overrides: c.update(overrides)
    return c
def evaluate_checklist(context):
    out=[]
    for item in CHECKLIST:
        out.append({"id":item["id"],"description":item["description"],"passed":bool(item["validator"](context))})
    return out
