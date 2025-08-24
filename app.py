# main.py
# RoofGuard AI — Roof Damage Analysis (Streamlit)
# v2.2.0 (launch-ready)
# - USA-only soft gate (state/ZIP)
# - "Report depth" (Recommended / Quick scan / Detailed assessment)
# - 4o-mini default; escalate worst image to 4o when needed
# - Stripe Checkout + Billing Portal (cancel/update card)
# - Supabase email-OTP login (require for subs/5-pack)
# - Session report history + optional Supabase Storage cloud history
# - PDF with embedded photos; Markdown download too
# - Email UI auto-hidden without creds
# - st.query_params (no experimental API)
# - Optional sample report link + promo note
# - HEIC support; basic image-quality hints

import os, io, time, base64, uuid, tempfile, re, json
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any, Union

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, ValidationError
from PIL import Image, ImageOps
import numpy as np

# HEIC/HEIF support
try:
    from pillow_heif import register_heif
    register_heif()
except Exception:
    pass

# PDF
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Payments (optional)
try:
    import stripe
except Exception:
    stripe = None

# Email (optional)
try:
    import yagmail
except Exception:
    yagmail = None

# OpenAI v1
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Supabase (optional, for login & storage)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# ===== Branding / Config =====
APP_NAME = "RoofGuard AI"
APP_VERSION = "2.2.0"

COMPANY_NAME = "RoofGuard AI"
COMPANY_EMAIL = "support@roofguard.ai"
COMPANY_PHONE = "+1 (555) 123-4567"
COMPANY_WEBSITE = "https://roofguard.ai"
COMPANY_LOGO_PATH = ""  # set via sidebar upload per-session

COOLDOWN_SECONDS = 10
LAUNCH_US_ONLY = True  # U.S. soft launch gate

DISCLAIMER = (
    "This is an AI-assisted visual assessment based on provided imagery. "
    "It is not a substitute for an on-site inspection by a licensed professional. "
    "Estimates are indicative and may vary by region, materials, and labor."
)

st.set_page_config(page_title=f"{APP_NAME} — Roof Damage Analysis", page_icon="🏠", layout="wide")

# ===== Secrets / Keys =====
load_dotenv(find_dotenv(), override=True)

def get_secret(env_key: str, secrets_key: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(env_key)
    if val:
        return val
    try:
        return st.secrets.get(secrets_key or env_key, default)
    except Exception:
        return default

def _mask(s: Optional[str]) -> str:
    if not s:
        return "MISSING"
    return f"{s[:4]}…{s[-4:]} (len={len(s)})"

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY",    "openai_api_key")
STRIPE_SECRET_KEY = get_secret("STRIPE_SECRET_KEY", "stripe_secret_key")
APP_BASE_URL      = get_secret("APP_BASE_URL",      "app_base_url") or "http://localhost:8501"

# Stripe Price IDs (create in Stripe and set in Secrets)
PRICE_SINGLE_ID   = get_secret("STRIPE_PRICE_SINGLE",  "stripe_price_single")
PRICE_5PACK_ID    = get_secret("STRIPE_PRICE_5PACK",   "stripe_price_5pack")
PRICE_MONTHLY_ID  = get_secret("STRIPE_PRICE_MONTHLY", "stripe_price_monthly")
PRICE_ANNUAL_ID   = get_secret("STRIPE_PRICE_ANNUAL",  "stripe_price_annual")

TERMS_URL         = get_secret("TERMS_URL",   "terms_url")   or "#"
PRIVACY_URL       = get_secret("PRIVACY_URL", "privacy_url") or "#"
ACCESS_CODE       = get_secret("ACCESS_CODE", "access_code") or ""
MAINT_MODE        = str(get_secret("MAINTENANCE_MODE", "maintenance_mode") or "0").strip().lower() in ("1","true","yes")

# Optional: sample links & promo note
SAMPLE_REPORT_URL = get_secret("SAMPLE_REPORT_URL", "sample_report_url")
PRICING_FAQ_URL   = get_secret("PRICING_FAQ_URL",   "pricing_faq_url")
PROMO_NOTE        = get_secret("PROMO_NOTE",        "promo_note")  # e.g., "Use code EARLY50 for 50% off your first report."

# Optional: default depth from env/secrets
DEFAULT_DEPTH = (os.getenv("DEFAULT_DEPTH") or get_secret("DEFAULT_DEPTH", "default_depth") or "Recommended").strip()

client: Optional[OpenAI] = None
if OPENAI_API_KEY and OpenAI:
    client = OpenAI(api_key=OPENAI_API_KEY)

if STRIPE_SECRET_KEY and stripe:
    stripe.api_key = STRIPE_SECRET_KEY

LIVE_MODE = bool(STRIPE_SECRET_KEY and STRIPE_SECRET_KEY.startswith("sk_live"))

# ===== Currency & Region =====
CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
FX_MAP = {("USD","USD"):1.0, ("USD","EUR"):0.92, ("USD","GBP"):0.78, ("USD","JPY"):155.0}
US_REGION_MULTIPLIER = {"National average":1.00, "Northeast":1.18, "Midwest":0.95, "South":0.92, "West":1.20}

def fmt_money(usd: float, currency: str) -> str:
    rate = FX_MAP.get(("USD", currency), 1.0)
    amount = usd * rate
    if currency == "JPY":
        amount = round(amount, -2)
    else:
        amount = round(amount)
    sym = CURRENCY_SYMBOL.get(currency, "")
    return f"{sym}{amount:,.0f}"

def fmt_money_both(usd: float, currency: str) -> str:
    main = fmt_money(usd, currency)
    return main if currency == "USD" else f"{main} (≈ ${usd:,.0f})"

def apply_region_multiplier(usd: float, region: str) -> float:
    return usd * US_REGION_MULTIPLIER.get(region, 1.0)

def fmt_currency_block(usd: float, currency: str, region: str) -> str:
    return fmt_money_both(apply_region_multiplier(usd, region), currency)

# ===== Entitlements (session only; persist later via DB/webhooks) =====
def init_entitlements():
    st.session_state.setdefault("free_total_grant", 3)
    st.session_state.setdefault("free_total_used", 0)
    st.session_state.setdefault("purchased_credits", 0)
    st.session_state.setdefault("unlimited_until", None)  # iso str
    st.session_state.setdefault("last_ts", 0.0)
    st.session_state.setdefault("emailed_reports", [])
    st.session_state.setdefault("stripe_customer_id", None)
    st.session_state.setdefault("user_email", None)  # from Supabase auth

def entitlements_status() -> Dict[str, Any]:
    u = st.session_state.get("unlimited_until")
    if isinstance(u, str):
        try:
            u = datetime.fromisoformat(u)
        except Exception:
            u = None
    return {
        "free_total_grant": st.session_state.get("free_total_grant", 3),
        "free_total_used":  st.session_state.get("free_total_used", 0),
        "purchased_credits": st.session_state.get("purchased_credits", 0),
        "unlimited_until": u,
    }

def can_generate_now() -> Tuple[bool, str]:
    s = entitlements_status()
    now = datetime.now()
    if s["unlimited_until"] and s["unlimited_until"] > now:
        return True, "Subscription active"
    if s["purchased_credits"] > 0:
        return True, "Using purchased credits"
    if s["free_total_used"] < s["free_total_grant"]:
        remain = s["free_total_grant"] - s["free_total_used"]
        return True, f"Free uses remaining: {remain}"
    return False, "Free limit reached — purchase a plan to continue."

def consume_credit():
    s = entitlements_status()
    now = datetime.now()
    if s["unlimited_until"] and s["unlimited_until"] > now:
        return
    if s["purchased_credits"] > 0:
        st.session_state["purchased_credits"] = s["purchased_credits"] - 1
        return
    st.session_state["free_total_used"] = s["free_total_used"] + 1

# ===== Data & AI =====
class RoofAssessment(BaseModel):
    damage_percentage: float = Field(ge=0, le=100)
    urgency: int = Field(ge=1, le=10)
    estimate_usd: float = Field(ge=0)
    analysis: str
    recommendation: str

SYSTEM_PROMPT = (
    "You are a professional roofing assessor. You CAN analyze the provided image. "
    "Return ONLY JSON with keys: damage_percentage (0-100), urgency (1-10), "
    "estimate_usd (>=0), analysis (2-3 sentences), recommendation (brief)."
)

def normalize_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    MAX_SIDE = 1600
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        r = MAX_SIDE / float(max(w, h))
        img = img.resize((int(w*r), int(h*r)))
    return img

def image_to_b64(img: Image.Image, quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # fallback: labeled lines (very defensive)
    dp = re.search(r"Damage\s*Percentage:\s*([\d.]+)", text, re.I)
    ur = re.search(r"Urgency\s*Level:\s*(\d+)", text, re.I)
    es = re.search(r"Repair\s*Estimate:\s*\$?\s*([\d.]+)", text, re.I)
    an = re.search(r"Detailed\s*Analysis:\s*(.+)", text, re.I)
    rc = re.search(r"Recommended\s*Action:\s*(.+)", text, re.I)
    if any([dp, ur, es, an, rc]):
        return {
            "damage_percentage": float(dp.group(1)) if dp else 0.0,
            "urgency": int(ur.group(1)) if ur else 5,
            "estimate_usd": float(es.group(1)) if es else 0.0,
            "analysis": (an.group(1).strip() if an else "No analysis."),
            "recommendation": (rc.group(1).strip() if rc else "No recommendation."),
        }
    return None

def call_openai_vision(img_b64: str, prefer: Optional[List[str]] = None) -> Tuple[Optional[RoofAssessment], str]:
    if not client:
        return None, "Error: OpenAI key missing; cannot analyze."
    if not img_b64 or len(img_b64) < 200:
        return None, "Error: encoded image appears empty."
    if prefer is None:
        prefer = ["gpt-4o-mini", "gpt-4o"]  # default mini-first

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this roof image and return ONLY JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        },
    ]

    last_err = "Unknown"
    for model_name in prefer:
        for use_resp_fmt in (True, False):
            try:
                kwargs = dict(model=model_name, messages=messages, temperature=0.2, max_tokens=450)
                if use_resp_fmt:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                data = try_extract_json(text)
                if data:
                    return RoofAssessment(**data), text
            except ValidationError as ve:
                return None, f"Validation error: {ve}"
            except Exception as e:
                last_err = str(e)
    return None, f"OpenAI error: {last_err}"

# ===== Report helpers =====
def new_report_id() -> str:
    return f"RG-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

def _save_temp_jpeg(img: Image.Image, quality: int = 92) -> str:
    im = img
    if im.mode != "RGB":
        im = im.convert("RGB")
    MAX_W = 2000
    if im.width > MAX_W:
        r = MAX_W / float(im.width)
        im = im.resize((int(im.width*r), int(im.height*r)))
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    im.save(tf, format="JPEG", quality=quality, optimize=True)
    tf.flush(); tf.close()
    return tf.name

def _save_temp_png_bytes(b: bytes) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    with open(tf.name, "wb") as f:
        f.write(b)
    return tf.name

def embed_img_md(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=85)
    b64 = base64.b64encode(b.getvalue()).decode()
    return f"![Analyzed Roof](data:image/jpeg;base64,{b64})"

def embed_logo_md(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"![{COMPANY_NAME} Logo](data:image/png;base64,{b64})"

def confidence_note(a: RoofAssessment) -> str:
    score = 0.6 + 0.4 * ((a.damage_percentage/100) * (a.urgency/10))
    score = max(0.0, min(1.0, score))
    if score >= 0.8: return "Confidence: High (clear visual indicators)."
    if score >= 0.6: return "Confidence: Medium (most indicators are consistent)."
    return "Confidence: Caution (limited cues; request additional photos)."

def has_email_creds() -> bool:
    return (
        yagmail is not None
        and bool(get_secret("EMAIL_USER", "email_user"))
        and bool(get_secret("EMAIL_PASS", "email_pass"))
    )

# Email helper (supports bcc)
def send_email_pdf(to_addr: str, subject: str, body_lines: List[str], pdf_bytes: bytes, filename: str, bcc: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    sender = get_secret("EMAIL_USER", "email_user")
    pwd    = get_secret("EMAIL_PASS", "email_pass")
    if not (yagmail and sender and pwd and to_addr):
        return False, "Email config missing (EMAIL_USER/EMAIL_PASS or yagmail)."
    try:
        tmpname = os.path.join(tempfile.gettempdir(), filename)
        with open(tmpname, "wb") as f:
            f.write(pdf_bytes)
        yag = yagmail.SMTP(user=sender, password=pwd)
        yag.send(to=to_addr, subject=subject, contents=body_lines, attachments=[tmpname], bcc=bcc or None)
        return True, None
    except Exception as e:
        return False, str(e)

# Markdown report
def build_markdown_report(
    report_id: str,
    ts: str,
    customer_name: str,
    address: str,
    notes: str,
    currency: str,
    region: str,
    assessments: List[Tuple[str, RoofAssessment, Image.Image]],
    summary_text: str,
    tech_name: str,
    tech_license: str,
    tech_phone: str,
) -> str:
    lines: List[str] = []
    logo_md = embed_logo_md(COMPANY_LOGO_PATH)
    if logo_md:
        lines.append(logo_md)
        lines.append("")

    lines.append(f"**{COMPANY_NAME}**  \n{COMPANY_EMAIL} • {COMPANY_PHONE} • {COMPANY_WEBSITE}")
    lines.append("")
    lines.append(f"# {APP_NAME} — Roof Damage Report")
    lines.append(f"**Report ID:** {report_id}")
    lines.append(f"**Date:** {ts}")
    lines.append(f"**Customer:** {customer_name or 'N/A'}")
    lines.append(f"**Address:** {address or 'N/A'}")
    lines.append(f"**Region (cost basis):** {region}")
    if notes:
        lines.append(f"**Notes:** {notes}")
    lines.append("")

    if tech_name or tech_license or tech_phone:
        lines.append("## Prepared By")
        if tech_name:
            lines.append(f"- **Technician:** {tech_name}")
        if tech_license:
            lines.append(f"- **License/Cert #:** {tech_license}")
        if tech_phone:
            lines.append(f"- **Contact:** {tech_phone}")
        lines.append("")

    if summary_text:
        lines.append("## Summary of Findings")
        lines.append(summary_text)
        lines.append("")

    for idx, (fname, a, img) in enumerate(assessments, start=1):
        lines.append(f"## Image {idx} — {fname}")
        lines.append(f"- **Damage %:** {a.damage_percentage:.1f}%")
        lines.append(f"- **Urgency:** {a.urgency}/10")
        lines.append(f"- **Estimate (adjusted):** {fmt_currency_block(a.estimate_usd, currency, region)}")
        lines.append(f"- **Analysis:** {a.analysis.strip()}")
        lines.append(f"- **Recommendation:** {a.recommendation.strip()}")
        lines.append(f"- {confidence_note(a)}")
        lines.append("")
        lines.append(embed_img_md(img))
        lines.append("")

    lines.append(f"---\n*{DISCLAIMER}*")
    return "\n".join(lines)

# PDF report
def build_pdf_report(
    report_id: str,
    ts: str,
    customer_name: str,
    address: str,
    notes: str,
    currency: str,
    region: str,
    assessments: List[Tuple[str, RoofAssessment, Image.Image]],
    summary_text: str,
    tech_name: str,
    tech_license: str,
    tech_phone: str,
    hires_photo: bool
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10))
    H1 = styles["Heading1"]; H1.fontName = "Helvetica-Bold"
    H2 = styles["Heading2"]; H2.fontName = "Helvetica-Bold"
    H3 = styles["Heading3"]; H3.fontName = "Helvetica-Bold"
    N  = styles["Normal"];   N.fontName  = "Helvetica"

    story: List[Any] = []
    story.append(Paragraph(f"{APP_NAME} — Roof Damage Report", H1))
    story.append(Spacer(1, 8))

    if COMPANY_LOGO_PATH and os.path.exists(COMPANY_LOGO_PATH):
        try:
            story.append(PDFImage(COMPANY_LOGO_PATH, width=160))
            story.append(Spacer(1, 6))
        except Exception:
            pass
    story.append(Paragraph(f"<b>{COMPANY_NAME}</b> • {COMPANY_EMAIL} • {COMPANY_PHONE} • {COMPANY_WEBSITE}", N))
    story.append(Spacer(1, 8))

    meta = (
        f"<b>Report ID:</b> {report_id}<br/>"
        f"<b>Date:</b> {ts}<br/>"
        f"<b>Customer:</b> {customer_name or 'N/A'}<br/>"
        f"<b>Address:</b> {address or 'N/A'}<br/>"
        f"<b>Region (cost):</b> {region}<br/>"
        f"<b>Notes:</b> {notes or 'N/A'}"
    )
    story.append(Paragraph(meta, N))
    story.append(Spacer(1, 10))

    prep = []
    if tech_name:    prep.append(f"<b>Prepared by:</b> {tech_name}")
    if tech_license: prep.append(f"<b>License/Cert #:</b> {tech_license}")
    if tech_phone:   prep.append(f"<b>Contact:</b> {tech_phone}")
    if prep:
        story.append(Paragraph("<br/>".join(prep), N))
        story.append(Spacer(1, 10))

    if summary_text:
        story.append(Paragraph("Summary of Findings", H2))
        story.append(Paragraph(summary_text, N))
        story.append(Spacer(1, 12))

    FRAME_W = doc.width
    TARGET_W = min(FRAME_W, 500 if hires_photo else 400)

    for idx, (fname, a, img) in enumerate(assessments, start=1):
        section: List[Any] = [Paragraph(f"Image {idx} — {fname}", H3)]
        try:
            jpg_path = _save_temp_jpeg(img, quality=92)
            w, h = img.size
            w = max(1, w)
            scale = TARGET_W / float(w)
            target_h = h * scale
            section.append(PDFImage(jpg_path, width=TARGET_W, height=target_h))
        except Exception as e:
            section.append(Paragraph(f"[Image could not be embedded: {e}]", N))
        details = (
            f"<b>Damage %:</b> {a.damage_percentage:.1f}%<br/>"
            f"<b>Urgency:</b> {a.urgency}/10<br/>"
            f"<b>Estimate (adjusted):</b> {fmt_currency_block(a.estimate_usd, currency, region)}<br/>"
            f"<b>Analysis:</b> {a.analysis}<br/>"
            f"<b>Recommendation:</b> {a.recommendation}<br/>"
            f"{confidence_note(a)}"
        )
        section.append(Paragraph(details, N))
        story.append(KeepTogether(section))
        story.append(Spacer(1, 12))

    story.append(Paragraph(DISCLAIMER, styles["Small"]))
    doc.build(story)
    return buf.getvalue()

# ===== Stripe helpers =====
def create_billing_portal(customer_id: str) -> Optional[str]:
    if not (stripe and STRIPE_SECRET_KEY and customer_id):
        return None
    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=APP_BASE_URL,
        )
        return portal.url
    except Exception as e:
        st.sidebar.error(f"Billing portal error: {e}")
        return None

# ===== Supabase helpers (auth + storage) =====
def get_supabase() -> Optional["Client"]:
    url = get_secret("SUPABASE_URL", "supabase_url")
    key = get_secret("SUPABASE_ANON_KEY", "supabase_anon_key")
    if not (create_client and url and key):
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def ensure_auth_ui() -> bool:
    """
    Renders login UI in the sidebar.
    Returns True if the user is signed in (session_state['user_email'] is set).
    """
    sb = get_supabase()
    st.sidebar.markdown("### 👤 Account")

    if not sb:
        # Login disabled if no Supabase keys; keep Singles usable
        if st.session_state.get("user_email"):
            st.sidebar.success(f"Signed in as {st.session_state['user_email']}")
            if st.sidebar.button("Sign out"):
                st.session_state.pop("user_email", None)
                st.rerun()
            return True
        st.sidebar.caption("Login disabled (no Supabase keys).")
        return False

    # Already signed in?
    if st.session_state.get("user_email"):
        st.sidebar.success(f"Signed in as {st.session_state['user_email']}")
        if st.sidebar.button("Sign out"):
            st.session_state.pop("user_email", None)
            st.rerun()
        return True

    # Not signed in → OTP flow
    email = st.sidebar.text_input("Email for login", key="login_email")
    col_s, col_v = st.sidebar.columns([1, 1])

    if col_s.button("Send code"):
        if not email or "@" not in email:
            st.sidebar.error("Enter a valid email.")
        else:
            try:
                sb.auth.sign_in_with_otp({"email": email})
                st.sidebar.info("Verification code sent. Check your inbox.")
            except Exception as e:
                st.sidebar.error(f"Send code failed: {e}")

    code = st.sidebar.text_input("Enter 6-digit code", key="login_code")
    if col_v.button("Verify"):
        if not (email and code):
            st.sidebar.error("Enter your email and the code.")
        else:
            try:
                sb.auth.verify_otp({"email": email, "token": code, "type": "email"})
                st.session_state["user_email"] = email.strip()
                st.sidebar.success("Logged in.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Verify failed: {e}")

    return False

def supabase_enabled() -> bool:
    return (
        get_supabase() is not None
        and bool(get_secret("SUPABASE_URL", "supabase_url"))
        and bool(get_secret("SUPABASE_ANON_KEY", "supabase_anon_key"))
    )

def save_report_to_supabase(
    user_email: str,
    report_id: str,
    md_text: str,
    pdf_bytes: bytes,
    meta: dict
) -> Tuple[bool, Union[str, Dict[str, str]]]:
    """
    Saves PDF, Markdown, and a small meta.json into a public storage bucket.
    Also maintains a simple per-user index.json for easy listing.
    """
    sb = get_supabase()
    if not (sb and supabase_enabled()):
        return False, "Supabase not configured."

    bucket = get_secret("SUPABASE_REPORTS_BUCKET", "supabase_reports_bucket") or "reports"
    user_folder = (user_email or "guest").replace("/", "_")
    prefix = f"{user_folder}/{report_id}"

    try:
        storage = sb.storage.from_(bucket)

        storage.upload(f"{prefix}/report.pdf", pdf_bytes, {"contentType": "application/pdf", "upsert": True})
        storage.upload(f"{prefix}/report.md", md_text.encode("utf-8"), {"contentType": "text/markdown", "upsert": True})
        storage.upload(f"{prefix}/meta.json", json.dumps(meta).encode("utf-8"), {"contentType": "application/json", "upsert": True})

        pdf_url = storage.get_public_url(f"{prefix}/report.pdf")
        md_url  = storage.get_public_url(f"{prefix}/report.md")

        # Maintain a lightweight index for quick listing
        idx_path = f"{user_folder}/index.json"
        try:
            existing = []
            try:
                raw = storage.download(idx_path)
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                existing = json.loads(text)
            except Exception:
                existing = []

            existing.append({
                "created_at": meta.get("created_at"),
                "report_id": report_id,
                "customer": meta.get("customer"),
                "address": meta.get("address"),
                "region": meta.get("region"),
                "depth": meta.get("depth"),
                "avg_damage": meta.get("avg_damage"),
                "max_urgency": meta.get("max_urgency"),
                "avg_estimate_usd": meta.get("avg_estimate_usd"),
                "pdf_url": pdf_url,
                "md_url": md_url,
            })

            storage.upload(idx_path, json.dumps(existing).encode("utf-8"),
                           {"contentType": "application/json", "upsert": True})
        except Exception:
            pass

        return True, {"pdf_url": pdf_url, "md_url": md_url}
    except Exception as e:
        return False, str(e)

def list_reports_from_supabase(user_email: str) -> List[Dict[str, Any]]:
    sb = get_supabase()
    if not (sb and supabase_enabled()):
        return []

    bucket = get_secret("SUPABASE_REPORTS_BUCKET", "supabase_reports_bucket") or "reports"
    storage = sb.storage.from_(bucket)
    user_folder = (user_email or "guest").replace("/", "_")
    idx_path = f"{user_folder}/index.json"

    try:
        raw = storage.download(idx_path)
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        data = json.loads(text)
        return sorted(data, key=lambda r: r.get("created_at", ""), reverse=True)[:100]
    except Exception:
        return []

# ===== UI =====
st.title(f"🏠 {APP_NAME}")
mode_label = "LIVE" if LIVE_MODE else "TEST"
st.caption(f"Professional Roof Damage Analysis & Reporting • v{APP_VERSION} • Stripe: {mode_label}")

# Optional header links
header_bits = []
if SAMPLE_REPORT_URL:
    header_bits.append(f"[View a sample report]({SAMPLE_REPORT_URL})")
if PRICING_FAQ_URL:
    header_bits.append(f"[Pricing & FAQs]({PRICING_FAQ_URL})")
if header_bits:
    st.markdown(" · ".join(header_bits))

init_entitlements()

# Sidebar — keys/base URL + logo uploader
st.sidebar.caption(f"OpenAI: {_mask(OPENAI_API_KEY)} | Stripe: {_mask(STRIPE_SECRET_KEY)}")
st.sidebar.caption(f"Base URL: {APP_BASE_URL}")

with st.sidebar.expander("Branding"):
    logo_file = st.file_uploader("Upload your logo (PNG)", type=["png"], accept_multiple_files=False, key="logo_up")
    if logo_file is not None:
        try:
            COMPANY_LOGO_PATH = _save_temp_png_bytes(logo_file.read())
            st.success("Logo uploaded.")
        except Exception as e:
            st.warning(f"Logo not set: {e}")

# Promo note
if PROMO_NOTE:
    st.sidebar.info(PROMO_NOTE)

# Auth UI (optional; require for subs)
logged_in = ensure_auth_ui()

# Country gate (USA soft launch)
st.markdown("### Customer / Site Info")
country = st.selectbox("Country", ["United States", "Other"], index=0)
if LAUNCH_US_ONLY and country != "United States":
    st.warning("We’re rolling out in the U.S. first. Join the waitlist and we’ll notify you when your country is supported.")
    waitlist_email = st.text_input("Your email for early access")
    st.stop()

us_col1, us_col2 = st.columns(2)
with us_col1:
    state = st.selectbox("State", [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
        "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
        "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
        "VA","WA","WV","WI","WY"
    ])
with us_col2:
    zipcode = st.text_input("ZIP code")

# Currency & Region
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"], index=0)
region = st.sidebar.selectbox("US region (cost basis)", list(US_REGION_MULTIPLIER.keys()), index=0)

# Report depth (customer-friendly)
depth_options = ["Recommended", "Quick scan", "Detailed assessment"]
default_idx = depth_options.index(DEFAULT_DEPTH) if DEFAULT_DEPTH in depth_options else 0
depth_choice = st.sidebar.selectbox(
    "Report depth",
    depth_options,
    index=default_idx,
    help="Quick = fastest triage. Detailed = deeper assessment for insurance."
)

# Advanced (admin-only) — hi-res toggle
passed_access = True
if ACCESS_CODE:
    st.sidebar.markdown("### 🔒 Access")
    provided = st.sidebar.text_input("Access code", type="password")
    passed_access = (provided == ACCESS_CODE)
    if not passed_access:
        st.sidebar.warning("Enter valid access code to enable advanced options.")

if passed_access:
    with st.sidebar.expander("Advanced"):
        hires_pdf = st.checkbox("High-res photo in PDF", value=False)
        st.caption("Use only when needed; files are larger and slower to generate.")
else:
    hires_pdf = False

# Email preferences (hidden if no creds)
st.session_state.setdefault("auto_email_me", False)
st.session_state.setdefault("auto_email_customer", False)
st.session_state.setdefault("bcc_me", False)
if has_email_creds():
    st.sidebar.markdown("### ✉️ Email preferences")
    st.session_state["auto_email_me"] = st.sidebar.checkbox(
        "Auto email me each report", value=st.session_state["auto_email_me"]
    )
    st.session_state["auto_email_customer"] = st.sidebar.checkbox(
        "Auto email customer (requires consent)", value=st.session_state["auto_email_customer"]
    )
    st.session_state["bcc_me"] = st.sidebar.checkbox(
        "BCC me on customer emails", value=st.session_state["bcc_me"]
    )
else:
    st.sidebar.caption("✉️ Email disabled (no EMAIL_USER/EMAIL_PASS set).")

# Maintenance & missing keys
if MAINT_MODE:
    st.sidebar.error("Maintenance mode ON — analysis disabled.")
if not OPENAI_API_KEY:
    st.sidebar.error("Missing OpenAI key.")

# Subscription status
st.sidebar.markdown("### 🔑 Subscription Status")
s = entitlements_status()
unlim_txt = "None"
if s["unlimited_until"] and s["unlimited_until"] > datetime.now():
    unlim_txt = s["unlimited_until"].strftime("%Y-%m-%d")
st.sidebar.write(
    f"- **Free uses total:** {s['free_total_grant'] - s['free_total_used']} remaining\n"
    f"- **Purchased credits:** {s['purchased_credits']}\n"
    f"- **Unlimited until:** {unlim_txt}"
)

# Plans / Checkout
st.sidebar.markdown("### 💳 Plans")
st.sidebar.write("• Pay-as-you-go or monthly • First 3 reports free • Cancel anytime")

plan_choice = st.sidebar.selectbox("Choose plan", ["Single $49", "5-Pack $199", "Monthly $299"])
annual = False
if "Monthly" in plan_choice:
    annual = st.sidebar.checkbox("Bill annually (2 months free)", value=False)

if st.sidebar.button("Checkout"):
    # Require login for 5-Pack & Monthly; allow Single without login
    needs_login = ("Monthly" in plan_choice) or ("5-Pack" in plan_choice)
    if needs_login and not logged_in:
        st.sidebar.warning("Please sign in (email code) before purchasing a subscription or 5-pack so your credits persist across devices.")
    else:
        if not STRIPE_SECRET_KEY or not stripe:
            st.sidebar.error("Stripe key missing — payments disabled.")
        else:
            if "Single" in plan_choice:
                price_id = PRICE_SINGLE_ID; mode = "payment"
            elif "5-Pack" in plan_choice:
                price_id = PRICE_5PACK_ID;  mode = "payment"
            else:
                price_id = PRICE_ANNUAL_ID if (annual and PRICE_ANNUAL_ID) else PRICE_MONTHLY_ID
                mode = "subscription"
            if not price_id:
                st.sidebar.error("Missing Stripe Price ID. Set it in Secrets.")
            else:
                try:
                    customer_email_for_stripe = st.session_state.get("user_email") if logged_in else None
                    session = stripe.checkout.Session.create(
                        mode=mode,
                        line_items=[{"price": price_id, "quantity": 1}],
                        success_url=f"{APP_BASE_URL}?checkout=success&plan={plan_choice}&annual={'1' if annual else '0'}&session_id={{CHECKOUT_SESSION_ID}}",
                        cancel_url=f"{APP_BASE_URL}?checkout=cancel",
                        allow_promotion_codes=True,
                        automatic_tax={"enabled": True},
                        customer_email=customer_email_for_stripe,
                    )
                    st.sidebar.markdown(f"[Complete Payment →]({session.url})")
                except Exception as e:
                    st.sidebar.error(f"Stripe error: {e}")

# Billing portal (self-serve cancel/update)
st.sidebar.markdown("### 🧾 Billing")
cust_id = st.session_state.get("stripe_customer_id")
if cust_id and stripe and STRIPE_SECRET_KEY:
    if st.sidebar.button("Manage subscription / billing"):
        url = create_billing_portal(cust_id)
        if url:
            st.sidebar.markdown(f"[Open Billing Portal →]({url})")
        else:
            st.sidebar.warning("Could not open portal. Try again.")
else:
    st.sidebar.caption("Billing portal available after your first successful purchase.")

# Handle success redirect & capture Stripe customer_id
qp = st.query_params
def _param(key, default=""):
    v = qp.get(key, default)
    if isinstance(v, list):
        return v[0] if v else default
    return v

if _param("checkout") == "success":
    plan = _param("plan")
    is_annual = _param("annual", "0") == "1"
    session_id = _param("session_id")

    if "5-Pack" in plan:
        st.session_state["purchased_credits"] = st.session_state.get("purchased_credits", 0) + 5
        st.success("Purchased 5 credits added to your account.")
    elif "Single" in plan:
        st.session_state["purchased_credits"] = st.session_state.get("purchased_credits", 0) + 1
        st.success("Purchased 1 credit added to your account.")
    elif "Monthly" in plan:
        days = 365 if is_annual else 30
        st.session_state["unlimited_until"] = (datetime.now() + timedelta(days=days)).isoformat()
        st.success(("Annual" if is_annual else "Monthly") + " unlimited activated.")

    if stripe and STRIPE_SECRET_KEY and session_id:
        try:
            cs = stripe.checkout.Session.retrieve(session_id, expand=["customer"])
            cust = cs.customer
            st.session_state["stripe_customer_id"] = cust.id if hasattr(cust, "id") else str(cust)
            st.info("Billing profile saved. You can manage your subscription from the sidebar.")
        except Exception as e:
            st.warning(f"Could not capture billing profile: {e}")

    st.query_params.clear()

# ===== Customer details =====
colA, colB = st.columns(2)
with colA:
    customer_name = st.text_input("Customer Name")
with colB:
    address = st.text_input("Property Address")
customer_email = st.text_input("Customer Email (optional)")
notes = st.text_area("Notes (optional)")

st.markdown("#### Your Company / Technician (shown on the report)")
colC, colD, colE = st.columns(3)
with colC:
    tech_name = st.text_input("Technician / Assessor Name", value="")
with colD:
    tech_license = st.text_input("License / Cert # (optional)", value="")
with colE:
    tech_phone = st.text_input("Contact Phone", value="")

with st.expander("📸 Photo tips (improves accuracy)"):
    st.markdown("""
- Capture 3–6 photos: ridge, valleys, flashing, gutters, obvious damage.
- Stand back to capture full sections; avoid heavy shadows.
- Add one close-up of the worst area if possible.
""")

# Terms acceptance
accept_terms = st.checkbox(
    f"I accept the [Terms]({TERMS_URL}) and [Privacy Policy]({PRIVACY_URL})",
    value=False
)

# Email consent + validation
def is_valid_email(s: str) -> bool:
    try:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s.strip()))
    except Exception:
        return False

email_consent = st.checkbox(
    "I confirm I have the customer's permission to email them this report",
    value=False, help="Required if you enable auto email to customer."
)

# Upload images
st.markdown("### Upload Roof Images")
uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG/HEIC)",
    type=["jpg", "jpeg", "png", "heic", "heif"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("**Preview**")
    cols = st.columns(4)
    for i, f in enumerate(uploaded_files):
        try:
            img_prev = Image.open(f); img_prev.thumbnail((250, 250))
            cols[i % 4].image(img_prev, caption=f.name, use_column_width=True)
        except Exception:
            cols[i % 4].warning(f"Cannot preview {f.name}")
        finally:
            f.seek(0)

# Cooldown
allow_time = True
if "last_ts" in st.session_state:
    elapsed = time.time() - st.session_state["last_ts"]
    if elapsed < COOLDOWN_SECONDS:
        allow_time = False
        st.info(f"Cooldown: wait {int(COOLDOWN_SECONDS - elapsed)}s before next run.")

# Analyze enablement
analyze_disabled = (
    MAINT_MODE or
    not allow_time or
    not accept_terms
)

# Model mapping for "Report depth"
def models_for_depth(depth: str, image_count: int) -> List[str]:
    if depth == "Quick scan":
        return ["gpt-4o-mini", "gpt-4o"]
    if depth == "Detailed assessment":
        return ["gpt-4o", "gpt-4o-mini"]
    # Recommended (auto): mini first for ≤3 photos; 4o first if more
    return ["gpt-4o-mini", "gpt-4o"] if image_count <= 3 else ["gpt-4o", "gpt-4o-mini"]

# Quick image quality guard
def quick_image_quality_note(img: Image.Image) -> Optional[str]:
    g = np.array(img.convert("L"), dtype=np.uint8)
    mean = float(g.mean()); var = float(g.var())
    if mean < 35:  return "Image looks very dark. Try better lighting or different angle."
    if mean > 220: return "Image looks overexposed. Try reducing glare or different time of day."
    if var < 200:  return "Image may be blurry/low-detail. Try a steadier shot or closer view."
    return None

if st.button("Analyze & Generate Report", type="primary", disabled=analyze_disabled):
    ok, reason = can_generate_now()
    if not ok:
        st.error(reason); st.stop()
    if not uploaded_files:
        st.error("Please upload at least one image."); st.stop()

    assessments: List[Tuple[str, RoofAssessment, Image.Image]] = []
    progress = st.progress(0.0); status = st.empty()

    prefer = models_for_depth(depth_choice, len(uploaded_files or []))

    for idx, f in enumerate(uploaded_files, start=1):
        status.write(f"Analyzing image {idx}/{len(uploaded_files)} …")
        try:
            pil = normalize_image(f)
            qnote = quick_image_quality_note(pil)
            if qnote:
                st.info(f"{f.name}: {qnote}")
            img_b64 = image_to_b64(pil)
            a, raw = call_openai_vision(img_b64, prefer=prefer)
            if a is None:
                st.error(f"Image '{f.name}' failed: {str(raw)[:400]}")
            else:
                assessments.append((f.name, a, pil))
        except Exception as e:
            st.error(f"Error with {f.name}: {e}")
        finally:
            progress.progress(idx/len(uploaded_files))
            f.seek(0)

    status.write("Analysis complete.")
    if not assessments:
        st.error("No valid analyses produced."); st.stop()

    # OPTIONAL: escalate one worst image if quick modes
    def needs_escalation(a: RoofAssessment) -> bool:
        if a.urgency >= 8 or a.damage_percentage >= 60:
            return True
        if len((a.analysis or "").split()) < 12:
            return True
        return False

    if depth_choice in ("Recommended", "Quick scan"):
        candidates = [(i, a) for i, (_, a, __) in enumerate(assessments) if needs_escalation(a)]
        if candidates:
            worst_i, _ = max(candidates, key=lambda t: (t[1].urgency, t[1].damage_percentage))
            fname, a_old, img = assessments[worst_i]
            try:
                a_new, _ = call_openai_vision(image_to_b64(img), prefer=["gpt-4o", "gpt-4o-mini"])
                if a_new:
                    assessments[worst_i] = (fname, a_new, img)
                    st.info(f"Did a deeper check on: {fname}")
            except Exception:
                pass

    consume_credit()

    avg_damage = sum(a.damage_percentage for _, a, __ in assessments) / len(assessments)
    max_urgency = max(a.urgency for _, a, __ in assessments)
    avg_est = sum(a.estimate_usd for _, a, __ in assessments) / len(assessments)
    adj_avg_est = apply_region_multiplier(avg_est, region)

    st.markdown("### Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Damage %", f"{avg_damage:.1f}%")
    c2.metric("Max Urgency", max_urgency)
    c3.metric("Avg Estimate", fmt_money_both(adj_avg_est, currency))

    st.write("### Job Details")
    st.write(f"- **Customer:** {customer_name or 'N/A'}")
    st.write(f"- **Address:** {address or 'N/A'}")
    st.write(f"- **State / ZIP:** {state} {zipcode or ''}")
    st.write(f"- **Region (cost basis):** {region}")
    st.write(f"- **Report depth:** {depth_choice}")
    if notes:
        st.write(f"- **Notes:** {notes}")

    summary_text = (
        f"For {customer_name or 'the client'} at {address or 'the property'}, the roof shows ~{avg_damage:.0f}% average damage "
        f"with a maximum urgency of {max_urgency}/10. Estimated average repair cost is "
        f"{fmt_money_both(adj_avg_est, currency)}. "
        "Next steps: (1) tarp/temporary waterproofing if leaks are present, "
        "(2) schedule contractor inspection within 7 days, "
        "(3) gather additional photos for insurer if filing a claim."
    )

    report_id = new_report_id()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    md_report = build_markdown_report(
        report_id, ts, customer_name, address, notes, currency, region,
        assessments, summary_text, tech_name, tech_license, tech_phone
    )
    st.markdown("---"); st.markdown("## Report Preview (Markdown)"); st.markdown(md_report)

    pdf_bytes = build_pdf_report(
        report_id, ts, customer_name, address, notes, currency, region,
        assessments, summary_text, tech_name, tech_license, tech_phone,
        hires_photo=hires_pdf
    )

    st.download_button(
        "⬇️ Download Report (Markdown)",
        data=md_report.encode("utf-8"),
        file_name=f"roof_report_{report_id}.md",
        mime="text/markdown"
    )
    st.download_button(
        "⬇️ Download Report (PDF)",
        data=pdf_bytes,
        file_name=f"roof_report_{report_id}.pdf",
        mime="application/pdf"
    )

    # --- Local session history (no setup required) ---
    st.session_state.setdefault("local_reports", [])
    st.session_state["local_reports"].append({
        "created_at": ts,
        "report_id": report_id,
        "customer": customer_name or "N/A",
        "address": address or "N/A",
        "region": region,
        "depth": depth_choice,
        "avg_damage": round(sum(a.damage_percentage for _, a, __ in assessments) / len(assessments), 1),
        "max_urgency": max(a.urgency for _, a, __ in assessments),
        "avg_estimate_usd": round(sum(a.estimate_usd for _, a, __ in assessments) / len(assessments), 0),
    })

    # --- Optional cloud history (Supabase) ---
    if supabase_enabled() and logged_in:
        meta = {
            "created_at": ts,
            "report_id": report_id,
            "customer": customer_name or "N/A",
            "address": address or "N/A",
            "region": region,
            "depth": depth_choice,
            "avg_damage": round(sum(a.damage_percentage for _, a, __ in assessments) / len(assessments), 1),
            "max_urgency": max(a.urgency for _, a, __ in assessments),
            "avg_estimate_usd": round(sum(a.estimate_usd for _, a, __ in assessments) / len(assessments), 0),
        }
        ok_cloud, info = save_report_to_supabase(
            st.session_state.get("user_email"), report_id, md_report, pdf_bytes, meta
        )
        if ok_cloud:
            st.success("Saved to cloud history.")
            st.markdown(f"[Open PDF]({info['pdf_url']})  ·  [Open Markdown]({info['md_url']})")
        else:
            st.info(f"Cloud save skipped: {info}")
    else:
        st.caption("Cloud history available when logged in with Supabase.")

    # Email preview + sending (if creds exist)
    if has_email_creds():
        with st.expander("✉️ Email content preview"):
            me_subject   = st.text_input("Subject (to me)", value=f"RoofGuard Report {report_id}", key=f"me_subj_{report_id}")
            me_body_txt  = st.text_area("Body (to me)",
                value=f"Report ID: {report_id}\nCustomer: {customer_name or 'N/A'}\nAddress: {address or 'N/A'}\nPDF attached.",
                key=f"me_body_{report_id}"
            )
            cust_subject = st.text_input("Subject (to customer)", value=f"Your Roof Report {report_id}", key=f"cust_subj_{report_id}")
            cust_body_txt= st.text_area("Body (to customer)",
                value=(f"Hi {customer_name or 'there'},\n"
                       "Attached is your roof damage report.\n"
                       "If you have questions, reply to this email.\n"
                       f"Prepared by: {tech_name or COMPANY_NAME}"),
                key=f"cust_body_{report_id}"
            )

        me_body   = me_body_txt.splitlines() if me_body_txt else []
        cust_body = cust_body_txt.splitlines() if cust_body_txt else []

        sender     = get_secret("EMAIL_USER", "email_user")
        pwd        = get_secret("EMAIL_PASS", "email_pass")
        default_to = get_secret("EMAIL_TO",  "email_to") or sender

        already_emailed = report_id in st.session_state["emailed_reports"]
        bcc_list = [default_to] if (st.session_state.get("bcc_me") and default_to) else []

        if not already_emailed:
            if bool(st.session_state.get("auto_email_me")) and sender and pwd and default_to:
                ok_send, err = send_email_pdf(default_to, me_subject, me_body, pdf_bytes, f"roof_report_{report_id}.pdf")
                if ok_send: st.success("Report emailed to you.")
                else:       st.warning(f"Email to you failed: {err}")

            if (bool(st.session_state.get("auto_email_customer"))
                and sender and pwd and customer_email
                and is_valid_email(customer_email)
                and email_consent):
                ok_send, err = send_email_pdf(customer_email, cust_subject, cust_body, pdf_bytes, f"roof_report_{report_id}.pdf", bcc=bcc_list)
                if ok_send: st.success(f"Report emailed to {customer_email}.")
                else:       st.warning(f"Email to customer failed: {err}")
            elif st.session_state.get("auto_email_customer") and customer_email and not email_consent:
                st.info("Customer auto-email skipped (consent not checked).")
            elif st.session_state.get("auto_email_customer") and customer_email and not is_valid_email(customer_email):
                st.info("Customer auto-email skipped (invalid email).")

            st.session_state["emailed_reports"].append(report_id)
        else:
            st.caption("Emails already sent for this report (skipping duplicates).")
    else:
        st.caption("✉️ Emailing is disabled. Download the PDF above.")

    st.session_state["last_ts"] = time.time()

# --- History panels ---
with st.expander("📁 Reports (this session)"):
    hist = st.session_state.get("local_reports", [])
    if not hist:
        st.caption("No reports yet.")
    else:
        for r in reversed(hist[-50:]):
            est = (f"${r['avg_estimate_usd']:,.0f}"
                   if isinstance(r.get("avg_estimate_usd"), (int, float)) else "—")
            st.write(
                f"**{r['created_at']}** • {r['customer']} — {r['address']} "
                f"• Depth: {r['depth']} • Avg damage: {r['avg_damage']}% "
                f"• Max urgency: {r['max_urgency']}/10 • Est: {est}"
            )

if supabase_enabled() and logged_in:
    with st.expander("☁️ Reports (cloud)"):
        rows = list_reports_from_supabase(st.session_state.get("user_email"))
        if not rows:
            st.caption("No cloud reports yet.")
        else:
            for r in rows:
                est = (f"${r['avg_estimate_usd']:,.0f}"
                       if isinstance(r.get("avg_estimate_usd"), (int, float)) else "—")
                pdf = r.get("pdf_url", "")
                md  = r.get("md_url", "")
                st.write(
                    f"**{r['created_at']}** • {r['customer']} — {r['address']} • Depth: {r['depth']} • "
                    f"Avg damage: {r['avg_damage']}% • Max urgency: {r['max_urgency']}/10 • Est: {est}  "
                    f"[PDF]({pdf}) · [MD]({md})"
                )

# Footer
st.markdown("---")
cols = st.columns(3)
cols[0].markdown(f"[Terms of Service]({TERMS_URL})")
cols[1].markdown(f"[Privacy Policy]({PRIVACY_URL})")
cols[2].markdown(f"© {datetime.now().year} {COMPANY_NAME} • v{APP_VERSION}")
