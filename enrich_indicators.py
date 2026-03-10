import json
import time
import argparse
import logging
import sys
import signal
from pathlib import Path

import requests


# ── Configuration ──────────────────────────────────────────────────────────────

WB_API_BASE   = "https://api.worldbank.org/v2/indicator"
API_BASE_URL  = "http://ollama.ollama-keda.svc.cluster.local:11434"
MODEL_NAME    = "gpt-oss:20b"
REQUEST_DELAY = 0.3


# ── ISIC Rev.4 Sections A–U ────────────────────────────────────────────────────

ISIC_SECTIONS = {
    "A": "Agriculture, Forestry and Fishing",
    "B": "Mining and Quarrying",
    "C": "Manufacturing",
    "D": "Electricity, Gas, Steam and Air Conditioning Supply",
    "E": "Water Supply, Sewerage, Waste Management and Remediation",
    "F": "Construction",
    "G": "Wholesale and Retail Trade; Repair of Motor Vehicles",
    "H": "Transportation and Storage",
    "I": "Accommodation and Food Service Activities",
    "J": "Information and Communication",
    "K": "Financial and Insurance Activities",
    "L": "Real Estate Activities",
    "M": "Professional, Scientific and Technical Activities",
    "N": "Administrative and Support Service Activities",
    "O": "Public Administration and Defence; Compulsory Social Security",
    "P": "Education",
    "Q": "Human Health and Social Work Activities",
    "R": "Arts, Entertainment and Recreation",
    "S": "Other Service Activities",
    "T": "Activities of Households as Employers",
    "U": "Activities of Extraterritorial Organisations and Bodies",
}

# WB topic → ISIC code mapping
WB_TOPIC_TO_ISIC = {
    "agriculture": "A",
    "climate change": "E",
    "economy": "K",
    "education": "P",
    "energy": "D",
    "environment": "E",
    "financial sector": "K",
    "gender": "O",
    "health": "Q",
    "infrastructure": "F",
    "labor": "N",
    "labour": "N",
    "poverty": "O",
    "private sector": "K",
    "public sector": "O",
    "science": "M",
    "social development": "Q",
    "social protection": "Q",
    "technology": "J",
    "trade": "G",
    "transport": "H",
    "urban development": "F",
    "water": "E",
}


def map_wb_topic_to_isic(topic_value: str) -> str:
    """Map a World Bank topic string to the closest ISIC section code."""
    lower = topic_value.lower().strip()
    for keyword, code in WB_TOPIC_TO_ISIC.items():
        if keyword in lower:
            return code
    return "S"   # fallback: Other Service Activities


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("enrich")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def checkpoint_path(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_name(p.stem + ".checkpoint.json"))


def load_checkpoint(ckpt_path: str, logger: logging.Logger) -> tuple:
    p = Path(ckpt_path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            enriched = data.get("enriched", [])
            next_idx = data.get("next_index", len(enriched))
            logger.info(f"Checkpoint found: {len(enriched)} items done, resuming from index {next_idx}")
            return enriched, next_idx
        except Exception as e:
            logger.warning(f"Could not read checkpoint ({e}), starting fresh")
    return [], 0


def save_checkpoint(ckpt_path: str, enriched: list, next_index: int,
                    logger: logging.Logger):
    tmp = ckpt_path + ".tmp"
    try:
        Path(tmp).write_text(
            json.dumps({"next_index": next_index, "enriched": enriched},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        Path(tmp).replace(Path(ckpt_path))
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def save_output(output_path: str, enriched: list, logger: logging.Logger):
    try:
        Path(output_path).write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Output saved → {output_path}  ({len(enriched)} records)")
    except Exception as e:
        logger.error(f"Failed to write output: {e}")


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
    except Exception:
        return False


def pull_model(model_name: str, api_base_url: str, logger: logging.Logger) -> bool:
    try:
        logger.info(f"Pulling model '{model_name}'...")
        response = requests.post(
            f"{api_base_url}/api/pull",
            headers={"Content-Type": "application/json"},
            json={"name": model_name, "stream": False},
            timeout=600
        )
        response.raise_for_status()
        logger.info("Model pulled successfully")
        return True
    except Exception as e:
        logger.error(f"Pull failed: {e}")
        return False


def ensure_model_available(model_name: str, api_base_url: str,
                           logger: logging.Logger) -> bool:
    if check_model_exists(model_name, api_base_url):
        logger.info(f"Model '{model_name}' is available")
        return True
    logger.warning(f"Model '{model_name}' not found, attempting to pull...")
    return pull_model(model_name, api_base_url, logger)


# ── World Bank API helper ──────────────────────────────────────────────────────

def fetch_wb_metadata(indicator_id: str, logger: logging.Logger) -> dict:
    """
    Returns domain as ISIC code (e.g. "Q") and context as
    "SubDomain – 2-3 line explanation" derived from the WB sourceNote.
    """
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
            indicator  = data[1][0]
            topics     = indicator.get("topics") or []
            topic_str  = topics[0]["value"].strip() if topics else ""
            source_note = indicator.get("sourceNote", "").strip()

            if topic_str or source_note:
                isic_code   = map_wb_topic_to_isic(topic_str)
                isic_label  = ISIC_SECTIONS.get(isic_code, "")

                # Build context: "TopicName – first ~2 sentences of sourceNote"
                sentences   = source_note.replace("  ", " ").split(". ")
                short_note  = ". ".join(sentences[:2]).strip()
                if short_note and not short_note.endswith("."):
                    short_note += "."

                sub_domain  = topic_str if topic_str else isic_label
                context     = f"{sub_domain} – {short_note}" if short_note else sub_domain

                logger.debug(f"  [WB API OK] ISIC={isic_code} topic='{topic_str}'")
                return {"domain": isic_code, "context": context}

        logger.debug(f"  [WB API] empty response for {indicator_id}")
    except Exception as exc:
        logger.debug(f"  [WB API error] {indicator_id}: {exc}")

    return None


# ── Deployed LLM fallback ──────────────────────────────────────────────────────

ISIC_LIST_FOR_PROMPT = "\n".join(
    f"  {code}: {label}" for code, label in ISIC_SECTIONS.items()
)

def fetch_llm_metadata(indicator_id: str, indicator_value: str,
                       logger: logging.Logger) -> dict:
    prompt = f"""You are a data analyst specialising in World Development Indicators (WDI).

Map the indicator below to the correct ISIC Rev.4 section and provide a short context.

ISIC Rev.4 sections:
{ISIC_LIST_FOR_PROMPT}

Rules:
- "domain" must be ONLY the single letter ISIC code (A through U) that best fits the indicator.
- "context" must follow this exact format:
    "<2-3 word sub-domain> – <2 to 3 sentence plain-English explanation of what the indicator measures>"
  Example: "Maternal Health – Tracks deaths related to pregnancy and childbirth per 100,000 live births. Used to assess quality of reproductive healthcare and identify gaps in prenatal and obstetric services."

Respond with ONLY a valid JSON object, no markdown, no extra text.

Indicator ID   : {indicator_id}
Indicator label: {indicator_value}

JSON response:"""

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=300
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]

        result  = json.loads(raw_text.strip())
        domain  = result.get("domain", "").strip().upper()
        context = result.get("context", "").strip()

        # Validate ISIC code
        if domain not in ISIC_SECTIONS:
            logger.warning(f"  [LLM] Invalid ISIC code '{domain}', defaulting to 'S'")
            domain = "S"

        logger.debug(f"  [LLM OK] ISIC={domain}")
        return {"domain": domain, "context": context}

    except Exception as exc:
        logger.error(f"  [LLM error] {indicator_id}: {exc}")

    return {"domain": "S", "context": ""}


# ── Main enrichment loop ───────────────────────────────────────────────────────

def enrich_indicators(input_path: str, output_path: str,
                      start_index: int, logger: logging.Logger):

    indicators = json.loads(Path(input_path).read_text(encoding="utf-8"))
    total = len(indicators)
    ckpt  = checkpoint_path(output_path)

    if start_index > 0:
        enriched, _ = load_checkpoint(ckpt, logger)
        enriched = enriched[:start_index]
        next_idx = start_index
        logger.info(f"Resuming from explicit index {start_index}")
    else:
        enriched, next_idx = load_checkpoint(ckpt, logger)

    if next_idx >= total:
        logger.info("All indicators already processed. Nothing to do.")
        save_output(output_path, enriched, logger)
        return

    logger.info(f"Total: {total} | Start: {next_idx} | Remaining: {total - next_idx}")

    logger.info("Checking model availability...")
    if not ensure_model_available(MODEL_NAME, API_BASE_URL, logger):
        logger.error("Model not available. Saving partial results and exiting.")
        save_output(output_path, enriched, logger)
        sys.exit(1)

    def _handle_signal(signum, frame):
        logger.warning(f"Signal {signum} received — saving partial results before exit...")
        save_output(output_path, enriched, logger)
        save_checkpoint(ckpt, enriched, current_idx, logger)
        logger.info(f"Saved {len(enriched)} records. Resume with --start-index {current_idx}")
        sys.exit(0)

    current_idx = next_idx
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    try:
        for i in range(next_idx, total):
            current_idx = i
            item      = indicators[i]
            ind_id    = item["id"]
            ind_value = item["value"]

            logger.info(f"[{i}/{total - 1}] {ind_id} – {ind_value}")

            meta = fetch_wb_metadata(ind_id, logger)
            source = "worldbank_api"

            if not meta:
                logger.info(f"  → WB API empty, falling back to LLM...")
                meta   = fetch_llm_metadata(ind_id, ind_value, logger)
                source = "llm_fallback"

            isic_code  = meta["domain"]
            isic_label = ISIC_SECTIONS.get(isic_code, "")
            logger.info(f"  → ISIC {isic_code}: {isic_label}")

            enriched.append({
                "index":   i,
                "id":      ind_id,
                "value":   ind_value,
                "domain":  f"{isic_label}-{isic_code}",
                "context": meta["context"],
                "_source": source
            })

            save_checkpoint(ckpt, enriched, i + 1, logger)
            time.sleep(REQUEST_DELAY)

    except Exception as e:
        logger.error(f"Unexpected error at index {current_idx}: {e}", exc_info=True)
        logger.error(f"Saving partial results. Resume with --start-index {current_idx}")
        save_output(output_path, enriched, logger)
        save_checkpoint(ckpt, enriched, current_idx, logger)
        sys.exit(1)

    save_output(output_path, enriched, logger)
    try:
        Path(ckpt).unlink(missing_ok=True)
        logger.info("Checkpoint file removed (run complete)")
    except Exception:
        pass

    logger.info(f"Done. {len(enriched)} indicators enriched.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich WDI indicators with ISIC domain and context."
    )
    parser.add_argument("input",
                        help="Path to input JSON file (list of {id, value} dicts)")
    parser.add_argument("-o", "--output", default="wdi-indicators-enriched.json",
                        help="Output JSON path (default: wdi-indicators-enriched.json)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Resume from this 0-based index (auto-detected from checkpoint if omitted)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--api-url", default=API_BASE_URL,
                        help=f"Deployed API base URL (default: {API_BASE_URL})")
    parser.add_argument("--log-file", default="enrich_indicators.log",
                        help="Log file path (default: enrich_indicators.log)")
    args = parser.parse_args()

    MODEL_NAME   = args.model
    API_BASE_URL = args.api_url

    logger = setup_logger(args.log_file)
    logger.info("=" * 60)
    logger.info("WDI Indicator Enrichment started")
    logger.info(f"  Input     : {args.input}")
    logger.info(f"  Output    : {args.output}")
    logger.info(f"  Log file  : {args.log_file}")
    logger.info(f"  Model     : {MODEL_NAME}")
    logger.info(f"  API URL   : {API_BASE_URL}")
    logger.info(f"  Start idx : {args.start_index if args.start_index else 'auto (checkpoint)'}")
    logger.info("=" * 60)

    enrich_indicators(args.input, args.output, args.start_index, logger)