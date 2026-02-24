import os
import json
import logging
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import openpyxl

# LangChain – document loading & splitting
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load .env file if python-dotenv is installed (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# AI client – provider-agnostic, configured via environment variables
import ai_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"

# ── Chunking config ────────────────────────────────────────────────────────
BRD_CHUNK_SIZE   = 4_000
EPIC_CHUNK_SIZE  = 2_000
CHUNK_OVERLAP    = 100

# ── Startup banner ─────────────────────────────────────────────────────────
_cfg = ai_client.get_config()
logger.info("=" * 60)
logger.info("  Project Dashboard — starting up")
logger.info("  AI endpoint : %s", _cfg["base_url"])
logger.info("  Model       : %s", _cfg["model"])
logger.info("  Timeout     : %ss    Max tokens: %s",
            _cfg["timeout"], _cfg["max_tokens"] or "provider default")
logger.info("=" * 60)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_prompt(filename: str) -> str:
    """Read a prompt file from the prompts/ directory."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def extract_excel_epics(path: str) -> str:
    """
    Parse an Excel file (.xlsx/.xls) and extract Epic items.

    Searches every sheet for a header row that contains both a 'Summary'
    and a 'Description' column (case-insensitive, any position).
    Returns a plain-text block with one entry per epic row:

        Epic 1 — <Summary>
        <Description>

    Rows where both Summary and Description are blank are skipped.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    entries: list[str] = []

    for sheet in wb.worksheets:
        # Scan rows to find the header row
        header_row_idx: int | None = None
        summary_col: int | None = None
        desc_col: int | None = None

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            # Look for the header row by searching for both target columns
            row_lower = [
                str(cell).strip().lower() if cell is not None else ""
                for cell in row
            ]
            has_summary = "summary" in row_lower
            has_desc    = "description" in row_lower
            if has_summary and has_desc:
                summary_col     = row_lower.index("summary")
                desc_col        = row_lower.index("description")
                header_row_idx  = row_idx
                break   # found header — stop scanning

        if header_row_idx is None:
            continue   # this sheet has no matching header; try next

        # Extract data rows (everything after the header)
        epic_num = 0
        for row in sheet.iter_rows(
            min_row=header_row_idx + 1, values_only=True
        ):
            summary = str(row[summary_col]).strip() if row[summary_col] is not None else ""
            desc    = str(row[desc_col]).strip()    if row[desc_col]    is not None else ""
            if not summary and not desc:
                continue
            epic_num += 1
            parts = [f"Epic {epic_num} — {summary}" if summary else f"Epic {epic_num}"]
            if desc:
                parts.append(desc)
            entries.append("\n".join(parts))

        if entries:
            break   # found data in this sheet; don't scan further sheets

    wb.close()

    if not entries:
        raise ValueError(
            "No 'Summary' and 'Description' columns found in the Excel file. "
            "Please ensure the file has a header row with those column names."
        )

    return "\n\n".join(entries)


def extract_text(tmp_path: str, ext: str, mime: str) -> str:
    """Load a document with the appropriate loader and return plain text."""
    if ext in (".xlsx", ".xls"):
        return extract_excel_epics(tmp_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)

    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)


def save_and_extract(file_storage) -> str:
    """Save a Flask FileStorage object to a temp file, extract its text, delete it."""
    if not file_storage or not file_storage.filename:
        return ""
    ext = os.path.splitext(file_storage.filename)[1].lower()
    mime = file_storage.mimetype or "application/octet-stream"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file_storage.save(tmp.name)
            tmp_path = tmp.name
        return extract_text(tmp_path, ext, mime)
    except Exception as exc:
        logger.warning("Could not extract text from %s: %s", file_storage.filename, exc)
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def chunk_text(text: str, chunk_size: int = EPIC_CHUNK_SIZE) -> list:
    """Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def ai_respond(system_prompt: str, conversation: list, user_message: str) -> str:
    """
    Delegate to ai_client.chat().
    conversation is a list of {"role": "user"|"assistant", "content": str}.
    """
    return ai_client.chat(system_prompt, conversation, user_message)


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from a model response."""
    raw = text.strip()
    if not raw:
        raise ValueError("AI returned an empty response instead of JSON.")
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
        
    # Fallback: find the outermost curly braces
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"AI returned invalid JSON. Response snippet: {raw[:200]}")


# System prompt used only for the BRD summarisation pass (terse, no JSON output)
_BRD_READER_SYSTEM = (
    "You are reading a Business Requirements Document section by section. "
    "For each section, extract the 2-3 most important technical scope items "
    "relevant for software effort estimation. "
    "Be extremely brief — one short line per item. No preamble, no numbering."
)


def run_estimation(
    system_prompt: str,
    brd_chunks: list,        # may be empty when BRD not uploaded
    epic_chunks: list,
    finalize_basic_msg: str,
    finalize_contextual_msg: str,
) -> tuple[dict, dict]:
    """
    Three-phase estimation:

    Phase 0 (BRD summarisation — only if brd_chunks provided):
      Each BRD chunk is sent stateless to _BRD_READER_SYSTEM.
      The model extracts 2-3 key scope bullets per chunk.
      All bullets are concatenated into a compact context block
      that is appended to system_prompt before Epic List processing.

    Phase 1 (Epic List reading):
      Intermediate epic chunks are sent stateless with the enriched
      system prompt — model acknowledges each with OK.

    Phase 2 (Epic List finalise — basic estimate):
      Final epic chunk + FINALIZE_BASIC → basic JSON.

    Phase 3 (Contextual follow-up):
      Reuses history from Phase 2 only; applies org rules.

    Returns (basic_result, contextual_result).
    """
    # Determine total number of AI calls for logging
    num_brd_calls = len(brd_chunks) if brd_chunks else 0
    num_epic_calls = len(epic_chunks) if epic_chunks else 0
    # +1 for final basic estimate, +1 for contextual follow-up
    total_calls = num_brd_calls + num_epic_calls + 2
    call_idx = 1

    # ── Phase 0: BRD map-reduce summarisation ─────────────────────────────
    enriched_system = system_prompt
    if brd_chunks:
        total_brd = len(brd_chunks)
        logger.info("BRD summarisation: %d chunks", total_brd)
        brd_bullets: list[str] = []
        for i, chunk in enumerate(brd_chunks):
            logger.info("API call %d/%d – BRD Section %d/%d", call_idx, total_calls, i + 1, total_brd)
            msg = (
                f"[BRD Section {i + 1} of {total_brd}]\n\n"
                f"{chunk}\n\n"
                "Summarise the key scope/technical items from this section "
                "(2-3 lines max)."
            )
            summary = ai_respond(_BRD_READER_SYSTEM, [], msg).strip()
            if summary:
                brd_bullets.append(summary)
            logger.info("BRD section %d/%d summarised (%d chars)",
                        i + 1, total_brd, len(summary))
            call_idx += 1

        if brd_bullets:
            brd_context = "\n".join(brd_bullets)
            enriched_system = (
                system_prompt
                + "\n\n────────────────────────────────────────────────\n"
                f"BRD CONTEXT ({total_brd} sections summarised):\n"
                f"{brd_context}\n"
                "────────────────────────────────────────────────"
            )
            logger.info("BRD context injected: %d chars total", len(brd_context))

    # ── Phase 1 & 2: Epic List (skipped if none uploaded) ─────────────────
    if epic_chunks:
        total_epic = len(epic_chunks)
        for i, chunk in enumerate(epic_chunks[:-1]):
            logger.info("API call %d/%d – Epic Chunk %d/%d", call_idx, total_calls, i + 1, total_epic)
            msg = (
                f"[Epic Chunk {i + 1} of {total_epic}]\n\n"
                f"{chunk}\n\n"
                f"Acknowledge with OK."
            )
            ai_respond(enriched_system, [], msg)
            logger.info("Epic chunk %d/%d sent", i + 1, total_epic)
            call_idx += 1

        final_chunk = epic_chunks[-1]
        final_msg = (
            f"[Epic Chunk {total_epic} of {total_epic} — FINAL]\n\n"
            f"{final_chunk}\n\n"
            f"---\n{finalize_basic_msg}"
        )
    else:
        # BRD-only mode: ask the model to estimate directly from the BRD context
        logger.info("No Epic List — estimating from BRD context only")
        final_msg = (
            "You have read the Business Requirements Document (summarised above).\n"
            f"---\n{finalize_basic_msg}"
        )

    logger.info("API call %d/%d – Finalize Basic", call_idx, total_calls)
    basic_text = ai_respond(enriched_system, [], final_msg)
    logger.info("Finalize sent | response: %d chars", len(basic_text))
    call_idx += 1

    if not basic_text:
        raise ValueError("No chunks were processed — Epic List was empty.")

    basic_result = _parse_json_response(basic_text)
    logger.info("Basic result: %s", basic_result)

    # ── Phase 3: contextual follow-up ──────────────────────────────────
    logger.info("API call %d/%d – Contextual Follow‑up", call_idx, total_calls)
    history = [
        {"role": "user",      "content": final_msg},
        {"role": "assistant", "content": basic_text},
    ]
    contextual_text = ai_respond(enriched_system, history, finalize_contextual_msg)
    logger.info("Contextual result: %s", contextual_text)
    contextual_result = _parse_json_response(contextual_text)
    logger.info("Contextual result: %s", contextual_result)

    return basic_result, contextual_result



def generate_insights(available_data: dict) -> list:
    """
    Generate 3-5 key insights via a single LM Studio call.
    """
    lines = []
    for label, md in available_data.items():
        sp = round(md * 1.3, 1)
        lines.append(f"  - {label}: {md} MD / {sp} SP")

    prompt = (
        "The following effort estimates are available for a software project:\n"
        + "\n".join(lines) + "\n\n"
        "Generate 3 to 5 key insights comparing these estimates. "
        "Focus on gaps between methods, accuracy implications, and actionable recommendations.\n"
        "Return ONLY a JSON array — no markdown fences, no explanation:\n"
        '[{"type": "green|yellow|blue", "title": "<short title>", "description": "<1-2 sentences>"}]'
    )

    system = "You are a project estimation analyst. Return only valid JSON."
    raw = ai_respond(system, [], prompt).strip()
    
    if not raw:
        raise ValueError("AI returned an empty response instead of insights JSON.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
        
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"AI returned invalid JSON for insights. Response snippet: {raw[:200]}")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/estimate", methods=["POST"])
def estimate():
    # ── 1. Collect file uploads ───────────────────────────────────────────
    brd_file          = request.files.get("brd")
    epic_list_file    = request.files.get("epic_list")
    jira_effort_file  = request.files.get("jira_effort")
    bitbucket_pr_file = request.files.get("bitbucket_pr")

    # ── 2. Collect text fields ────────────────────────────────────────────
    stakeholder_md_raw = request.form.get("stakeholder_md", "").strip()
    try:
        stakeholder_md_val = float(stakeholder_md_raw) if stakeholder_md_raw else None
    except ValueError:
        stakeholder_md_val = None

    # TODO: re-enable once Epic List is mandatory again
    # if not epic_list_file or not epic_list_file.filename:
    #     if brd_file and brd_file.filename:
    #         return jsonify({"error": "An Epic List is required to estimate. Please upload the Epic List (the BRD alone is not sufficient)."}), 400
    #     return jsonify({"error": "Please upload an Epic List to begin estimation."}), 400

    try:
        # ── 3. Extract Epic List text (optional for now) ──────────────────
        epic_list_text = ""
        if epic_list_file and epic_list_file.filename:
            epic_list_text = save_and_extract(epic_list_file)
            if not epic_list_text.strip():
                return jsonify({"error": "Could not extract any text from the Epic List document."}), 400

        # ── 4. Extract BRD as optional background context ─────────────────
        brd_text = save_and_extract(brd_file) if (brd_file and brd_file.filename) else ""

        # Must have at least one source
        if not epic_list_text.strip() and not brd_text.strip():
            return jsonify({"error": "Please upload a BRD or an Epic List to begin estimation."}), 400

        # ── 5. Extract other optional supplementary files ─────────────────
        jira_effort_text = save_and_extract(jira_effort_file)
        bitbucket_text   = save_and_extract(bitbucket_pr_file)
        has_jira_effort  = bool(jira_effort_text)
        has_bitbucket    = bool(bitbucket_text)

        # ── 6. Chunk BRD (optional) and Epic List for AI processing ───────────
        brd_chunks  = chunk_text(brd_text, chunk_size=BRD_CHUNK_SIZE) if brd_text.strip() else []
        epic_chunks = chunk_text(epic_list_text) if epic_list_text.strip() else []
        if brd_chunks:
            logger.info("BRD: %d chars → %d chunks (size=%d)",
                        len(brd_text), len(brd_chunks), BRD_CHUNK_SIZE)
        if epic_chunks:
            logger.info("Epic List: %d chars → %d chunks", len(epic_list_text), len(epic_chunks))

        # ── 7. Load prompts ───────────────────────────────────────────────
        basic_prompt          = load_prompt("basic_prompt.txt")
        advanced_instructions = load_prompt("advanced_instructions.txt")


        # ── 8. Build finalize messages ────────────────────────────────────
        n = len(epic_chunks)
        FINALIZE_BASIC = (
            f"This is the final Epic List chunk ({n} of {n}). You have now read all items.\n"
            "Produce a basic effort estimate (no organizational context) for the TOTAL project.\n"
            "Return ONLY this JSON — no markdown fences, no explanation:\n"
            '{"project_name": "<inferred name>", "ai_basic": {"md": <integer>, "sp": <float>}}'
        )

        # Contextual uses org rules only — actuals handled separately
        FINALIZE_CONTEXTUAL = (
            "You have already produced a basic estimate. Now apply ADVANCED organizational rules.\n\n"
            f"{advanced_instructions}\n\n"
            "Return ONLY this JSON — no markdown fences, no explanation:\n"
            '{"project_name": "<inferred name>", "ai_contextual": {"md": <integer>, "sp": <float>}}'
        )

        # ── 9. Estimation: BRD summarisation + Epic List pass ─────────────────
        logger.info("Starting estimation…")
        basic_result, contextual_result = run_estimation(
            basic_prompt, brd_chunks, epic_chunks, FINALIZE_BASIC, FINALIZE_CONTEXTUAL
        )

        # ── 8. Build available_data dict for insight generation ───────────
        project_name = (
            contextual_result.get("project_name")
            or basic_result.get("project_name", "")
        )

        ai_basic      = basic_result.get("ai_basic", {})
        ai_contextual = contextual_result.get("ai_contextual", {})
        # TODO: extract jira_scope / jira_actual / bitbucket directly from uploaded files
        jira_scope = None
        jira_actual = None
        bitbucket = None

        # Collect whichever estimates we actually have for insight generation
        available_data = {}
        if stakeholder_md_val is not None:
            available_data["Stakeholder Estimate"] = stakeholder_md_val
        if ai_basic.get("md") is not None:
            available_data["AI Estimate (Basic)"] = ai_basic["md"]
        if ai_contextual.get("md") is not None:
            available_data["AI Estimate (Contextual)"] = ai_contextual["md"]
        if jira_scope and jira_scope.get("md") is not None:
            available_data["Jira Scope"] = jira_scope["md"]
        if jira_actual and jira_actual.get("md") is not None:
            available_data["Jira Actual Effort"] = jira_actual["md"]
        if bitbucket and bitbucket.get("md") is not None:
            available_data["Bitbucket Coding Effort"] = bitbucket["md"]

        # ── 9. Generate insights from available data ──────────────────────
        insights = []
        if len(available_data) >= 2:
            logger.info("Generating insights from: %s", list(available_data.keys()))
            try:
                insights = generate_insights(available_data)
            except Exception as exc:
                logger.warning("Insight generation failed: %s", exc)

        # ── 10. Compute accuracy (only if jira_actual known) ─────────────
        accuracy = {}
        actual_md = jira_actual.get("md") if jira_actual else None
        if actual_md:
            def _acc(md):
                return round(max(0, min(100, (1 - abs(md - actual_md) / actual_md) * 100)), 1)
            if stakeholder_md_val is not None:
                accuracy["stakeholder"] = _acc(stakeholder_md_val)
            if ai_basic.get("md"):
                accuracy["ai-basic"] = _acc(ai_basic["md"])
            if ai_contextual.get("md"):
                accuracy["ai-contextual"] = _acc(ai_contextual["md"])
            if jira_scope and jira_scope.get("md"):
                accuracy["jira-scope"] = _acc(jira_scope["md"])

        # ── 11. Build response (only include fields with real data) ───────
        response = {
            "project_name":  project_name,
            "ai_basic":      ai_basic,
            "ai_contextual": ai_contextual,
            "insights":      insights,
        }
        if accuracy:
            response["accuracy"] = accuracy
        if jira_scope:
            response["jira_scope"] = jira_scope
        if jira_actual:
            response["jira_actual"] = jira_actual
        if bitbucket:
            response["bitbucket"] = bitbucket

        return jsonify(response)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI returned invalid JSON: {e}"}), 500
    except Exception as e:
        logger.exception("Estimation failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)

