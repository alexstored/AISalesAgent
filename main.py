import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI

# =========================
# Load environment
# =========================
load_dotenv()

# --- Required for HubSpot flows ---
HUBSPOT_PRIVATE_APP_TOKEN = os.getenv("HUBSPOT_PRIVATE_APP_TOKEN")

# --- Required for AI advice (server-side only) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Your existing Render service that stores current call state ---
CT_STATE_API_BASE = os.getenv(
    "CT_STATE_API_BASE",
    "https://new-ct-number-pull.onrender.com",
).rstrip("/")

# --- CORS (allow your Vercel app + localhost) ---
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://*.vercel.app",
)

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "StoredAI/1.0 (+https://stored.ai)",
)

# Website intel settings
INTEL_MAX_CHARS = int(os.getenv("INTEL_MAX_CHARS", "6000"))
INTEL_USE_AI = os.getenv("INTEL_USE_AI", "true").lower() == "true"

# AI settings
AI_MODEL = os.getenv("AI_MODEL", "gpt-4.1-mini")

# Optional: HubSpot note association type id (varies by portal/object model)
HUBSPOT_NOTE_TO_CONTACT_ASSOC_TYPE_ID = os.getenv("HUBSPOT_NOTE_TO_CONTACT_ASSOC_TYPE_ID")

# =========================
# App + CORS
# =========================
app = FastAPI(title="StoredAI Backend", version="0.1.0")

# Convert CSV -> list; keep simple wildcard support for vercel preview
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Clients
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Helpers
# =========================
def normalize_phone(raw: str) -> Tuple[str, str, str]:
    """
    Returns: (e164_like, digits_only, last10)
    """
    raw = (raw or "").strip()
    digits = re.sub(r"\D+", "", raw)
    if raw.startswith("+"):
        e164 = "+" + digits
    else:
        # best effort: if UK numbers coming without '+', you might need better logic later
        e164 = "+" + digits if digits else ""

    last10 = digits[-10:] if len(digits) >= 10 else digits
    return e164, digits, last10


def require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing required server env var: {name}")
    return value


async def http_get_text(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text


def extract_site_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = meta["content"].strip()

    # Grab headings and some body text
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = h.get_text(" ", strip=True)
        if txt and len(txt) <= 120:
            headings.append(txt)
        if len(headings) >= 20:
            break

    body_text = soup.get_text(" ", strip=True)
    body_text = re.sub(r"\s+", " ", body_text)

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if meta_desc:
        parts.append(f"META_DESCRIPTION: {meta_desc}")
    if headings:
        parts.append("HEADINGS: " + " | ".join(headings[:20]))
    if body_text:
        parts.append("BODY: " + body_text[:INTEL_MAX_CHARS])

    out = "\n".join(parts)
    return out[:INTEL_MAX_CHARS]


def safe_domain(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


# =========================
# HubSpot API helpers
# =========================
def hubspot_headers() -> Dict[str, str]:
    token = require_env("HUBSPOT_PRIVATE_APP_TOKEN", HUBSPOT_PRIVATE_APP_TOKEN)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


async def hubspot_search_contact_by_phone(phone_raw: str) -> Optional[Dict[str, Any]]:
    """
    Attempts multiple matching strategies across phone fields.
    Returns the first match's object summary (id + properties).
    """
    e164, digits, last10 = normalize_phone(phone_raw)
    if not digits:
        return None

    url = "https://api.hubapi.com/crm/v3/objects/contacts/search"

    # HubSpot search supports OR across filterGroups, AND within group.
    # We'll OR across fields & match styles.
    filter_groups = []

    # Exact matches
    for prop in ["phone", "mobilephone"]:
        if e164:
            filter_groups.append({"filters": [{"propertyName": prop, "operator": "EQ", "value": e164}]})
        filter_groups.append({"filters": [{"propertyName": prop, "operator": "EQ", "value": digits}]})

    # Token contains (useful if formatting differs)
    for prop in ["phone", "mobilephone"]:
        filter_groups.append({"filters": [{"propertyName": prop, "operator": "CONTAINS_TOKEN", "value": last10}]})

    payload = {
        "filterGroups": filter_groups,
        "properties": [
            "firstname",
            "lastname",
            "email",
            "phone",
            "mobilephone",
            "company",
            "website",
            "address",
            "city",
            "zip",
            "country",
        ],
        "limit": 5,
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(url, headers=hubspot_headers(), json=payload)
        if r.status_code == 401:
            raise HTTPException(status_code=500, detail="HubSpot auth failed (check HUBSPOT_PRIVATE_APP_TOKEN).")
        r.raise_for_status()
        data = r.json()

    results = data.get("results", [])
    return results[0] if results else None


async def hubspot_get_contact(contact_id: str) -> Dict[str, Any]:
    url = f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}"
    params = {
        "properties": ",".join(
            [
                "firstname",
                "lastname",
                "email",
                "phone",
                "mobilephone",
                "company",
                "website",
                "address",
                "city",
                "zip",
                "country",
                "hs_object_id",
            ]
        )
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url, headers=hubspot_headers(), params=params)
        r.raise_for_status()
        return r.json()


async def hubspot_get_associated_company_id(contact_id: str) -> Optional[str]:
    url = f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}/associations/companies"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url, headers=hubspot_headers())
        r.raise_for_status()
        data = r.json()
    results = data.get("results", [])
    return results[0].get("id") if results else None


async def hubspot_get_company(company_id: str) -> Dict[str, Any]:
    url = f"https://api.hubapi.com/crm/v3/objects/companies/{company_id}"
    params = {
        "properties": ",".join(
            [
                "name",
                "domain",
                "website",
                "phone",
                "address",
                "city",
                "zip",
                "country",
                "industry",
                "description",
                "hs_object_id",
            ]
        )
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url, headers=hubspot_headers(), params=params)
        r.raise_for_status()
        return r.json()


async def hubspot_create_note(note_body_html: str) -> Dict[str, Any]:
    """
    Creates a Note object in HubSpot. Association to contact is optional and may
    require associationTypeId (portal-specific).
    """
    url = "https://api.hubapi.com/crm/v3/objects/notes"
    payload = {
        "properties": {
            "hs_note_body": note_body_html,
            "hs_timestamp": str(int(time.time() * 1000)),
        }
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(url, headers=hubspot_headers(), json=payload)
        r.raise_for_status()
        return r.json()


async def hubspot_associate_note_to_contact(note_id: str, contact_id: str) -> None:
    """
    Optional. Requires association type id, which can vary.
    If you don't know it yet, skip association for now and just store note ID.
    """
    assoc_type_id = HUBSPOT_NOTE_TO_CONTACT_ASSOC_TYPE_ID
    if not assoc_type_id:
        return

    url = (
        f"https://api.hubapi.com/crm/v3/objects/notes/{note_id}/associations/"
        f"contacts/{contact_id}/{assoc_type_id}"
    )
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.put(url, headers=hubspot_headers())
        r.raise_for_status()


# =========================
# Models (API contracts)
# =========================
class PullCallResponse(BaseModel):
    agent_id: str
    active: bool
    state: Optional[str] = None
    external_number: Optional[str] = None
    call_uuid: Optional[str] = None
    age_seconds: Optional[int] = None
    updated_at: Optional[int] = None


class HubSpotContact(BaseModel):
    id: str
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    mobilephone: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None


class HubSpotCompany(BaseModel):
    id: str
    name: Optional[str] = None
    domain: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None
    country: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None


class PullCallEnrichedResponse(BaseModel):
    call: PullCallResponse
    hubspot_contact: Optional[HubSpotContact] = None
    hubspot_company: Optional[HubSpotCompany] = None
    website: Optional[str] = None  # derived best website/domain to scan


class WebsiteIntelRequest(BaseModel):
    url: str = Field(..., min_length=3)


class WebsiteIntelResponse(BaseModel):
    url: str
    bullets: List[str]
    extracted_preview: Optional[str] = None  # helpful for debugging


class AdviceRequest(BaseModel):
    call_notes: str = Field("", description="Free-form notes captured during the call")
    website_bullets: List[str] = Field(default_factory=list)
    provider: Optional[str] = None
    pricing_context: Optional[str] = None  # e.g. "AOV £45, MCT £10, terminals 2"
    extra_context: Optional[str] = None


class AdviceResponse(BaseModel):
    bullets: List[str]


class SaveNotesRequest(BaseModel):
    contact_id: Optional[str] = None
    call_uuid: Optional[str] = None
    notes: str = Field(..., min_length=1)


class SaveNotesResponse(BaseModel):
    ok: bool
    note_id: Optional[str] = None
    associated: bool = False


# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "storedai-backend",
        "ct_state_api_base": CT_STATE_API_BASE,
        "hubspot_configured": bool(HUBSPOT_PRIVATE_APP_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
    }


@app.get("/call/current", response_model=PullCallResponse)
async def get_current_call(agent_id: str = Query(..., min_length=1)):
    """
    Pull current call state from your existing Render call-state service.
    """
    url = f"{CT_STATE_API_BASE}/agents/{agent_id}/current-call"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return PullCallResponse(**data)


@app.get("/call/pull", response_model=PullCallEnrichedResponse)
async def pull_call_and_enrich(agent_id: str = Query(..., min_length=1)):
    """
    1) Pull current call from CT_STATE_API_BASE
    2) If active + external_number -> search HubSpot contact
    3) Pull full contact + associated company
    4) Return combined payload for your UI
    """
    call = await get_current_call(agent_id)

    if not call.active or not call.external_number:
        return PullCallEnrichedResponse(call=call)

    require_env("HUBSPOT_PRIVATE_APP_TOKEN", HUBSPOT_PRIVATE_APP_TOKEN)

    # Search contact by phone
    found = await hubspot_search_contact_by_phone(call.external_number)
    if not found:
        # Return call only; UI can show number + "no match"
        return PullCallEnrichedResponse(call=call, website=None)

    contact_id = found.get("id")
    contact_obj = await hubspot_get_contact(contact_id)
    props = contact_obj.get("properties", {}) or {}

    contact = HubSpotContact(
        id=contact_id,
        firstname=props.get("firstname"),
        lastname=props.get("lastname"),
        email=props.get("email"),
        phone=props.get("phone"),
        mobilephone=props.get("mobilephone"),
        website=props.get("website"),
        address=props.get("address"),
        city=props.get("city"),
        zip=props.get("zip"),
        country=props.get("country"),
    )

    # Pull company if associated
    company = None
    website = contact.website or None

    company_id = await hubspot_get_associated_company_id(contact_id)
    if company_id:
        company_obj = await hubspot_get_company(company_id)
        cprops = company_obj.get("properties", {}) or {}
        company = HubSpotCompany(
            id=company_id,
            name=cprops.get("name"),
            domain=cprops.get("domain"),
            website=cprops.get("website"),
            address=cprops.get("address"),
            city=cprops.get("city"),
            zip=cprops.get("zip"),
            country=cprops.get("country"),
            industry=cprops.get("industry"),
            description=cprops.get("description"),
        )
        website = website or company.website or company.domain

    if website:
        website = safe_domain(website)

    return PullCallEnrichedResponse(
        call=call,
        hubspot_contact=contact,
        hubspot_company=company,
        website=website,
    )


@app.post("/intel/website", response_model=WebsiteIntelResponse)
async def website_intelligence(req: WebsiteIntelRequest):
    """
    Fetches and extracts website text, returns 3-4 bullets describing business type/offer.
    If INTEL_USE_AI=true and OpenAI key present, it will AI-summarise.
    Otherwise returns heuristic bullets (title/desc/headings).
    """
    url = safe_domain(req.url)

    headers = {"User-Agent": USER_AGENT}
    try:
        html = await http_get_text(url, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Website fetch failed: {str(e)}")

    extracted = extract_site_text(html)

    # Non-AI bullets fallback
    fallback_bullets = []
    for line in extracted.splitlines():
        if line.startswith("TITLE:"):
            fallback_bullets.append(line.replace("TITLE:", "").strip())
        if line.startswith("META_DESCRIPTION:"):
            fallback_bullets.append(line.replace("META_DESCRIPTION:", "").strip())
        if line.startswith("HEADINGS:"):
            hs = line.replace("HEADINGS:", "").strip().split("|")
            hs = [h.strip() for h in hs if h.strip()]
            fallback_bullets.extend(hs[:2])
        if len(fallback_bullets) >= 4:
            break
    fallback_bullets = [b for b in fallback_bullets if b][:4]
    if not fallback_bullets:
        fallback_bullets = ["Business website detected.", "No clear description found.", "Try scanning the About page."]

    if not (INTEL_USE_AI and openai_client):
        return WebsiteIntelResponse(url=url, bullets=fallback_bullets[:4], extracted_preview=extracted[:600])

    prompt = f"""
You are an analyst. From the website text below, infer the business type and what they sell/do.
Return 3-4 concise bullet points (max 14 words each). No fluff.

Return ONLY valid JSON like:
{{"bullets": ["...", "..."]}}

WEBSITE_TEXT:
{extracted}
""".strip()

    try:
        resp = openai_client.responses.create(
            model=AI_MODEL,
            input=prompt,
            temperature=0.2,
        )
        raw = (resp.output_text or "").strip()
        parsed = json.loads(raw)
        bullets = parsed.get("bullets", [])
        if not isinstance(bullets, list) or not bullets:
            raise ValueError("AI returned no bullets")
        bullets = [str(b).strip().lstrip("-•").strip() for b in bullets if str(b).strip()]
        bullets = bullets[:4]
        return WebsiteIntelResponse(url=url, bullets=bullets, extracted_preview=extracted[:600])
    except Exception:
        # Fall back gracefully
        return WebsiteIntelResponse(url=url, bullets=fallback_bullets[:4], extracted_preview=extracted[:600])


@app.post("/ai/advice", response_model=AdviceResponse)
async def get_sales_advice(req: AdviceRequest):
    """
    Uses OpenAI to produce 2-3 actionable sales bullets based on:
    - call notes
    - website intelligence bullets
    - provider/pricing context
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server.")

    website_bits = "\n".join([f"- {b}" for b in req.website_bullets[:8]]) if req.website_bullets else "- (none)"
    provider = req.provider or "(unknown)"
    pricing = req.pricing_context or "(not provided)"
    extra = req.extra_context or ""

    prompt = f"""
You are a top-performing UK SME payments/FX sales coach.
Produce 2-3 sharp, practical bullets for what the agent should say NEXT on this call.

Inputs:
- Current provider: {provider}
- Pricing context: {pricing}
- Website intel:
{website_bits}
- Call notes:
{req.call_notes}

Optional extra:
{extra}

Rules:
- Bullets should be actionable and specific (not generic).
- Mention 1 strong question the agent should ask.
- Keep each bullet <= 18 words.
Return ONLY valid JSON like:
{{"bullets": ["...", "..."]}}
""".strip()

    try:
        resp = openai_client.responses.create(
            model=AI_MODEL,
            input=prompt,
            temperature=0.2,
        )
        raw = (resp.output_text or "").strip()
        parsed = json.loads(raw)
        bullets = parsed.get("bullets", [])
        if not isinstance(bullets, list) or not bullets:
            raise ValueError("AI returned no bullets")
        bullets = [str(b).strip().lstrip("-•").strip() for b in bullets if str(b).strip()]
        return AdviceResponse(bullets=bullets[:3])
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="AI response was not valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.post("/hubspot/notes", response_model=SaveNotesResponse)
async def save_notes_to_hubspot(req: SaveNotesRequest):
    """
    Saves notes into HubSpot as a Note object.
    Association to contact is optional (depends on associationTypeId availability).
    """
    require_env("HUBSPOT_PRIVATE_APP_TOKEN", HUBSPOT_PRIVATE_APP_TOKEN)

    # Store as simple HTML note body. You can enrich format later.
    body_html = f"""
<p><strong>Call UUID:</strong> {req.call_uuid or ""}</p>
<p><strong>Notes:</strong></p>
<p>{(req.notes or "").replace("\\n", "<br/>")}</p>
""".strip()

    note = await hubspot_create_note(body_html)
    note_id = note.get("id")

    associated = False
    if req.contact_id and note_id:
        try:
            await hubspot_associate_note_to_contact(note_id, req.contact_id)
            associated = bool(HUBSPOT_NOTE_TO_CONTACT_ASSOC_TYPE_ID)
        except Exception:
            # association is optional; don't fail the whole request
            associated = False

    return SaveNotesResponse(ok=True, note_id=note_id, associated=associated)