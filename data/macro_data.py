"""Helpers for obtaining free macroeconomic data and simple news sentiment.

This module uses only the standard library so that it works out of the box.
If a FRED API key is provided in the environment (FRED_API_KEY) it will
fetch the requested series.  Otherwise it will return an empty list.

News sentiment is extremely naive: it downloads an RSS feed and counts
positive/negative words from a small lexicon.  The aim is not accuracy but
rather to have *something* that tracks headlines.
"""
import os
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

POSITIVE_WORDS = {"gain", "rise", "bull", "beat", "optimist"}
NEGATIVE_WORDS = {"drop", "fall", "bear", "miss", "pessimist"}


def fetch_fred_series(series_id: str, start_date: str = None, end_date: str = None):
    """Return list of observations from FRED (requires environment key).

    Args:
        series_id: e.g. "GDP" or "CPALTT01USM657N".
        start_date: YYYY-MM-DD (optional)
        end_date: YYYY-MM-DD (optional)

    Returns:
        list of dict with keys 'date' and 'value'.  Empty list on failure.
    """
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return []
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    if start_date: params["observation_start"] = start_date
    if end_date: params["observation_end"] = end_date
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.load(r)
            obs = data.get("observations", [])
            return [{"date": o.get("date"), "value": o.get("value")} for o in obs]
    except Exception:
        return []


def simple_rss_sentiment(rss_url: str):
    """Return a crude sentiment score based on RSS headlines.

    Score is (#positive - #negative) / total_articles.
    """
    try:
        with urllib.request.urlopen(rss_url, timeout=5) as r:
            xml = r.read()
    except Exception:
        return 0.0
    try:
        root = ET.fromstring(xml)
    except Exception:
        return 0.0
    texts = []
    # typical RSS has channel/item/title
    for title in root.findall('.//item/title'):
        if title is not None and title.text:
            texts.append(title.text.lower())
    if not texts:
        return 0.0
    pos = neg = 0
    for t in texts:
        words = set(t.split())
        if words & POSITIVE_WORDS:
            pos += 1
        if words & NEGATIVE_WORDS:
            neg += 1
    return (pos - neg) / max(1, len(texts))
