import json
import time
import argparse
import requests
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────

WB_API_BASE   = "https://api.worldbank.org/v2/indicator"
API_BASE_URL  = "http://ollama-keda.mobiusdtaas.ai"
MODEL_NAME    = "gpt-oss:20b"
REQUEST_DELAY = 0.3


# ── Model availability helpers ─────────────────────────────────────────────────

def check_model_exists(model_name: str, api_base_url: str) -> bool:
    try:
        response = requests.get(
            f"{api_base_url}/api/tags",
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(m.get("name") == model_name for m in models)
    except Exception as e:
        print(f"  [Model check error] {e}")
        return False


def pull_model(model_name: str, api_base_url: str) -> bool:
    try:
        print(f"  → Pulling model '{model_name}'...")
        response = requests.post(
            f"{api_base_url}/api/pull",
            headers={"Content-Type": "application/json"},
            json={"name": model_name, "stream": False},
            timeout=600
        )
        response.raise_for_status()
        print(f"    Model pulled successfully")
        return True
    except Exception as e:
        print(f"  [Pull error] {e}")
        return False


def ensure_model_available(model_name: str, api_base_url: str) -> bool:
    if check_model_exists(model_name, api_base_url):
        print(f"  ✓ Model '{model_name}' is available")
        return True
    print(f"  Model '{model_name}' not found, attempting to pull...")
    return pull_model(model_name, api_base_url)


# ── World Bank API helper ──────────────────────────────────────────────────────

def fetch_wb_metadata(indicator_id: str) -> dict | None:
    url = f"{WB_API_BASE}/{indicator_id}?format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if (
            isinstance(data, list)
            and len(data) == 2
            and isinstance(data[1], list)
            and data[1]
        ):
            indicator = data[1][0]
            topics = indicator.get("topics") or []
            domain  = topics[0]["value"].strip() if topics else ""
            context = indicator.get("sourceNote", "").strip()

            if domain or context:
                return {"domain": domain, "context": context}
    except Exception as exc:
        print(f"  [WB API error] {indicator_id}: {exc}")

    return None


# ── Deployed LLM fallback ──────────────────────────────────────────────────────

def fetch_llm_metadata(indicator_id: str, indicator_value: str) -> dict:
    prompt = f"""You are a data analyst specialising in World Development Indicators (WDI).

Given the indicator below, respond with ONLY a valid JSON object (no markdown, no extra text)
containing exactly two keys:
  "domain"  - the high-level subject area (e.g. "Health", "Education", "Poverty & Inequality")
  "context" - one or two sentences explaining what the indicator measures

Indicator ID   : {indicator_id}
Indicator label: {indicator_value}

JSON response:"""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        # Strip markdown fences if present
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]

        result = json.loads(raw_text.strip())
        return {
            "domain":  result.get("domain", "").strip(),
            "context": result.get("context", "").strip()
        }

    except Exception as exc:
        print(f"  [LLM error] {indicator_id}: {exc}")

    return {"domain": "", "context": ""}


# ── Main enrichment loop ───────────────────────────────────────────────────────

def enrich_indicators(input_path: str, output_path: str):
    indicators = json.loads(Path(input_path).read_text(encoding="utf-8"))
    total = len(indicators)

    print("Checking model availability...")
    if not ensure_model_available(MODEL_NAME, API_BASE_URL):
        print("✗ Model not available. Exiting.")
        return

    print(f"\nProcessing {total} indicators...\n")
    enriched = []

    for i, item in enumerate(indicators, 1):
        ind_id    = item["id"]
        ind_value = item["value"]
        print(f"[{i}/{total}] {ind_id} – {ind_value}")

        # 1. Try World Bank API
        meta = fetch_wb_metadata(ind_id)
        source = "worldbank_api"

        # 2. Fallback to deployed LLM
        if not meta:
            print(f"  → API returned nothing, using LLM fallback...")
            meta = fetch_llm_metadata(ind_id, ind_value)
            source = "llm_fallback"

        enriched.append({
            "id":      ind_id,
            "value":   ind_value,
            "domain":  meta["domain"],
            "context": meta["context"],
            "_source": source
        })

        time.sleep(REQUEST_DELAY)

    Path(output_path).write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n✓ Wrote {len(enriched)} enriched indicators → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich WDI indicators with domain and context."
    )
    parser.add_argument("input", help="Path to input JSON file (list of {id, value} dicts)")
    parser.add_argument("-o", "--output", default="wdi-indicators-enriched.json",
                        help="Output JSON path (default: wdi-indicators-enriched.json)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--api-url", default=API_BASE_URL,
                        help=f"Deployed API base URL (default: {API_BASE_URL})")
    args = parser.parse_args()

    MODEL_NAME   = args.model
    API_BASE_URL = args.api_url

    enrich_indicators(args.input, args.output)