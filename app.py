# app.py
# RoofGuard AI — Roof Damage Analysis (Streamlit)
# OpenAI v1 SDK (no deprecated openai.File). Base64 data URLs for vision.
# Includes: multi-image, JSON validation, PDF/Markdown export, EUR/GBP/JPY,
# subscription status (MVP), high-res PDF toggle, robust PDF image embedding.

import os, io, time, base64, uuid, tempfile
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, ValidationError
from PIL import Image, ImageOps

# PDF
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Payments (optional)
try:
    import stripe
except Exception:
    stripe = None

# OpenAI v1
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# Branding / Company Config
# =========================
APP_NAME = "RoofGuard AI"           # rename if you prefer: "StormShield Reports", "RoofCheck Pro"
APP_VERSION = "1.5.1"

COMPANY_NAME = "RoofGuard AI"
COMPANY_EMAIL = "support@roofguard.ai"
COMPANY_PHONE = "+1 (555) 123-4567"
COMPANY_WEBSITE = "https://roofguard.ai"
COMPANY_LOGO_PATH = ""  # e.g., "logo.png" in the same folder (optional)

COOLDOWN_SECONDS = 10
DEBUG = False

DISCLAIMER = (
    "This is an AI-assisted visual assessment based on provided imagery. "
    "It is not a substitute for an on-site inspection by a licensed professional. "
    "Estimates are indicative and may vary by region, materials, and labor."
)

# Page config must be first Streamlit command
st.set_page_config(page_title=f"{APP_NAME} — Roof Damage Analysis", page_icon="🏠", layout="wide")

# =========================
# Secrets / Keys loading
# =========================
load_dotenv(find_dotenv(), override=True)

def get_secret(env_key: str, secrets_key: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(env_key)
    if val:
        return val
    try:
        return st.secrets.get(secrets_key or env_key, default)
    except Exception:
        return default

OPENAI_API_KEY    = get_secret("OPENAI_API_KEY", "openai_api_key")
STRIPE_SECRET_KEY = get_secret("STRIPE_SECRET_KEY", "stripe_secret_key")
APP_BASE_URL      = get_secret("APP_BASE_URL", "app_base_url") or "http://localhost:8501"

def _mask(s: Optional[str]) -> str:
    if not s: return "MISSING"
    return f"{s[:4]}…{s[-4:]} (len={len(s)})"

client: Optional[OpenAI] = None
if OPENAI_API_KEY and OpenAI:
    client = OpenAI(api_key=OPENAI_API_KEY)

if STRIPE_SECRET_KEY and stripe:
    stripe.api_key = STRIPE_SECRET_KEY

# =========================
# Currency
# =========================
CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
FX_MAP = {
    ("USD", "USD"): 1.0,
    ("USD", "EUR"): 0.92,   # placeholder rates; update as needed
    ("USD", "GBP"): 0.78,
    ("USD", "JPY"): 155.0,
}

def fmt_money(usd: float, currency: str) -> str:
    rate = FX_MAP.get(("USD", currency), 1.0)
    amount = usd * rate
    if currency == "JPY":
        amount = round(amount, -2)  # nearest ¥100
    else:
        amount = round(amount)      # nearest unit
    sym = CURRENCY_SYMBOL.get(currency, "")
    return f"{sym}{amount:,.0f}"

def fmt_money_both(usd: float, currency: str) -> str:
    main = fmt_money(usd, currency)
    if currency == "USD":
        return main
    return f"{main} (≈ ${usd:,.0f})"

# =========================
# Subscriptions & Credits (MVP in-memory)
# =========================
def init_entitlements():
    st.session_state.setdefault("free_total_grant", 3)   # free lifetime per session
    st.session_state.setdefault("free_total_used", 0)
    st.session_state.setdefault("purchased_credits", 0)  # single/bundle credits
    st.session_state.setdefault("unlimited_until", None) # datetime iso str
    st.session_state.setdefault("last_ts", 0.0)

def entitlements_status() -> Dict[str, Any]:
    unlimited_until = st.session_state.get("unlimited_until")
    if isinstance(unlimited_until, str):
        try:
            unlimited_until = datetime.fromisoformat(unlimited_until)
        except Exception:
            unlimited_until = None
    return {
        "free_total_grant": st.session_state.get("free_total_grant", 3),
        "free_total_used":  st.session_state.get("free_total_used", 0),
        "purchased_credits": st.session_state.get("purchased_credits", 0),
        "unlimited_until": unlimited_until,
    }

def can_generate_now() -> Tuple[bool, str]:
    s = entitlements_status()
    now = datetime.now()
    if s["unlimited_until"] and s["unlimited_until"] > now:
        return True, "Subscription active"
    if s["purchased_credits"] > 0:
        return True, "Using purchased credits"
    if s["free_total_used"] < s["free_total_grant"]:
        return True, f"Free uses remaining: {s['free_total_grant'] - s['free_total_used']}"
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

# =========================
# Data models & helpers
# =========================
class RoofAssessment(BaseModel):
    damage_percentage: float = Field(ge=0, le=100)
    urgency: int = Field(ge=1, le=10)
    estimate_usd: float = Field(ge=0)
    analysis: str
    recommendation: str

SYSTEM_PROMPT = (
    "You are a professional roofing assessor. You CAN analyze the provided image. "
    "Return ONLY JSON with keys: damage_percentage (0-100), urgency (1-10), "
    "estimate_usd (>=0), analysis (2-3 sentences), recommendation (brief). "
    "No extra text. If uncertain, state assumptions in 'analysis'."
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
    import json, re
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
    # fallback: handle labeled format
    dp = re.search(r"Damage\s*Percentage:\s*([\d.]+)\s*%?", text, re.I)
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
            "recommendation": (rc.group(1).strip() if rc else "No recommendation.")
        }
    return None

def call_openai_vision(img_b64: str) -> Tuple[Optional[RoofAssessment], str]:
    if not client:
        return None, "Error: OpenAI key missing; cannot analyze."
    if not img_b64 or len(img_b64) < 200:
        return None, "Error: encoded image appears empty."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this roof image and return ONLY JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ],
        },
    ]

    def _try(model_name: str, use_resp_fmt: bool):
        try:
            kwargs = dict(model=model_name, messages=messages, temperature=0.2, max_tokens=450)
            if use_resp_fmt:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip(), None
        except Exception as e:
            return None, str(e)

    text, err = _try("gpt-4o", True)
    if not text:
        text, err = _try("gpt-4o", False)
    if not text:
        text, err = _try("gpt-4o-mini", False)
    if not text:
        return None, f"OpenAI error: {err}"

    data = try_extract_json(text)
    if not data:
        return None, f"Could not parse JSON from model output.\n\nRaw:\n{text}"
    try:
        return RoofAssessment(**data), text
    except ValidationError as ve:
        return None, f"Validation error: {ve}\n\nRaw:\n{text}"

def fmt_currency_block(usd: float, currency: str) -> str:
    return fmt_money_both(usd, currency)

def new_report_id() -> str:
    return f"RG-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

def embed_img_md(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=85)
    b64 = base64.b64encode(b.getvalue()).decode()
    return f"![Analyzed Roof](data:image/jpeg;base64,{b64})"

def embed_logo_md(path: str) -> str:
    if not path or not os.path.exists(path): return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"![{COMPANY_NAME} Logo](data:image/png;base64,{b64})"

def confidence_note(a: RoofAssessment) -> str:
    score = 0.6 + 0.4 * ((a.damage_percentage/100) * (a.urgency/10))
    score = max(0.0, min(1.0, score))
    if score >= 0.8: return "Confidence: High (clear visual indicators)."
    if score >= 0.6: return "Confidence: Medium (most indicators are consistent)."
    return "Confidence: Caution (limited cues; request additional photos)."

# Save PIL image to a temp JPEG path (most reliable for ReportLab)
def _save_temp_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Normalize → downscale huge images → save to temp JPEG → return path."""
    im = img
    if im.mode != "RGB":
        im = im.convert("RGB")
    MAX_W = 2000
    if im.width > MAX_W:
        r = MAX_W / float(im.width)
        im = im.resize((int(im.width * r), int(im.height * r)))
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    im.save(tf, format="JPEG", quality=quality, optimize=True)
    tf.flush(); tf.close()
    return tf.name

def build_markdown_report(
    report_id: str, ts: str, customer_name: str, address: str, notes: str, currency: str,
    assessments: List[Tuple[str, RoofAssessment, Image.Image]], summary_text: str,
    tech_name: str, tech_license: str, tech_phone: str
) -> str:
    lines: List[str] = []
    # Company header
    logo_md = embed_logo_md(COMPANY_LOGO_PATH)
    if logo_md:
        lines.append(logo_md); lines.append("")
    lines.append(f"**{COMPANY_NAME}**  \n{COMPANY_EMAIL} • {COMPANY_PHONE} • {COMPANY_WEBSITE}")
    lines.append("")
    lines.append(f"# {APP_NAME} — Roof Damage Report")
    lines.append(f"**Report ID:** {report_id}")
    lines.append(f"**Date:** {ts}")
    lines.append(f"**Customer:** {customer_name or 'N/A'}")
    lines.append(f"**Address:** {address or 'N/A'}")
    if notes: lines.append(f"**Notes:** {notes}")
    lines.append("")
    # Prepared By
    if tech_name or tech_license or tech_phone:
        lines.append("## Prepared By")
        if tech_name:    lines.append(f"- **Technician:** {tech_name}")
        if tech_license: lines.append(f"- **License/Cert #:** {tech_license}")
        if tech_phone:   lines.append(f"- **Contact:** {tech_phone}")
        lines.append("")
    # Summary
    if summary_text:
        lines.append("## Summary of Findings")
        lines.append(summary_text); lines.append("")
    # Per image
    for idx, (fname, a, img) in enumerate(assessments, start=1):
        lines.append(f"## Image {idx} — {fname}")
        lines.append(f"- **Damage %:** {a.damage_percentage:.1f}%")
        lines.append(f"- **Urgency:** {a.urgency}/10")
        lines.append(f"- **Estimate ({currency}):** {fmt_currency_block(a.estimate_usd, currency)}")
        lines.append(f"- **Analysis:** {a.analysis.strip()}")
        lines.append(f"- **Recommendation:** {a.recommendation.strip()}")
        lines.append(f"- {confidence_note(a)}")
        lines.append("")
        lines.append(embed_img_md(img))
        lines.append("")
    lines.append(f"---\n*{DISCLAIMER}*")
    return "\n".join(lines)

def build_pdf_report(
    report_id: str,
    ts: str,
    customer_name: str,
    address: str,
    notes: str,
    currency: str,
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

    # Title
    story.append(Paragraph(f"{APP_NAME} — Roof Damage Report", H1))
    story.append(Spacer(1, 8))

    # Company header (logo + contacts)
    if COMPANY_LOGO_PATH and os.path.exists(COMPANY_LOGO_PATH):
        try:
            story.append(PDFImage(COMPANY_LOGO_PATH, width=160))
            story.append(Spacer(1, 6))
        except Exception:
            pass
    story.append(Paragraph(f"<b>{COMPANY_NAME}</b> • {COMPANY_EMAIL} • {COMPANY_PHONE} • {COMPANY_WEBSITE}", N))
    story.append(Spacer(1, 8))

    # Meta
    meta = (
        f"<b>Report ID:</b> {report_id}<br/>"
        f"<b>Date:</b> {ts}<br/>"
        f"<b>Customer:</b> {customer_name or 'N/A'}<br/>"
        f"<b>Address:</b> {address or 'N/A'}<br/>"
        f"<b>Notes:</b> {notes or 'N/A'}"
    )
    story.append(Paragraph(meta, N))
    story.append(Spacer(1, 8))

    # Prepared By
    prep = []
    if tech_name:    prep.append(f"<b>Prepared by:</b> {tech_name}")
    if tech_license: prep.append(f"<b>License/Cert #:</b> {tech_license}")
    if tech_phone:   prep.append(f"<b>Contact:</b> {tech_phone}")
    if prep:
        story.append(Paragraph("<br/>".join(prep), N))
        story.append(Spacer(1, 10))

    # Summary
    if summary_text:
        story.append(Paragraph("Summary of Findings", H2))
        story.append(Paragraph(summary_text, N))
        story.append(Spacer(1, 10))

    # Layout widths
    FRAME_W = doc.width
    TARGET_W = min(FRAME_W, 500 if hires_photo else 400)

    # Per-image sections
    for idx, (fname, a, img) in enumerate(assessments, start=1):
        section: List[Any] = [Paragraph(f"Image {idx} — {fname}", H3)]

        # Save to temp JPEG and compute scaled height (keeps aspect ratio)
        try:
            jpg_path = _save_temp_jpeg(img, quality=92)
            w, h = img.size
            if w == 0:
                w = 1
            scale = TARGET_W / float(w)
            target_h = h * scale
            section.append(PDFImage(jpg_path, width=TARGET_W, height=target_h))
        except Exception as e:
            section.append(Paragraph(f"[Image could not be embedded: {e}]", N))

        details = (
            f"<b>Damage %:</b> {a.damage_percentage:.1f}%<br/>"
            f"<b>Urgency:</b> {a.urgency}/10<br/>"
            f"<b>Estimate ({currency}):</b> {fmt_currency_block(a.estimate_usd, currency)}<br/>"
            f"<b>Analysis:</b> {a.analysis}<br/>"
            f"<b>Recommendation:</b> {a.recommendation}<br/>"
            f"{confidence_note(a)}"
        )
        section.append(Paragraph(details, N))
        story.append(KeepTogether(section))
        story.append(Spacer(1, 12))

    # Disclaimer
    story.append(Paragraph(DISCLAIMER, styles["Small"]))

    # Build & return bytes
    doc.build(story)
    return buf.getvalue()

# =========================
# UI
# =========================
st.title(f"🏠 {APP_NAME}")
st.caption(f"Professional Roof Damage Analysis & Reporting • v{APP_VERSION}")

# Initialize entitlements
init_entitlements()

# Sidebar: Keys / Currency / Quality
st.sidebar.caption(f"Keys → OpenAI: {_mask(OPENAI_API_KEY)} | Stripe: {_mask(STRIPE_SECRET_KEY)}")
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"], index=0)
hires_pdf = st.sidebar.checkbox("High-res photo in PDF", value=False)

# Sidebar: Subscription Status
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

# Sidebar: Plans / Checkout (optional)
st.sidebar.markdown("### 💳 Plans")
st.sidebar.write("Upgrade to continue after free credits are used.")
plan_choice = st.sidebar.selectbox("Choose plan", ["Single $49", "5-Pack $199", "Monthly $299/mo"])
if st.sidebar.button("Checkout"):
    if not STRIPE_SECRET_KEY or not stripe:
        st.sidebar.error("Stripe key missing — payments disabled.")
    else:
        desc = plan_choice
        amount = 4900 if "Single" in plan_choice else (19900 if "5-Pack" in plan_choice else 29900)
        mode = "payment" if "Monthly" not in plan_choice else "subscription"
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                mode=mode,
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": f"{APP_NAME} — {desc}"},
                        "unit_amount": amount
                    },
                    "quantity": 1
                }],
                success_url=f"{APP_BASE_URL}?checkout=success&plan={desc}",
                cancel_url=f"{APP_BASE_URL}?checkout=cancel",
                allow_promotion_codes=True,
                automatic_tax={"enabled": True},
            )
            st.sidebar.markdown(f"[Complete Payment →]({session.url})")
        except Exception as e:
            st.sidebar.error(f"Stripe error: {e}")

# Handle naive "post-purchase" (MVP) via query params (demo only)
qp = st.experimental_get_query_params()
if qp.get("checkout") == ["success"]:
    plan = qp.get("plan", [""])[0]
    if "5-Pack" in plan:
        st.session_state["purchased_credits"] += 5
        st.success("Purchased 5 credits added to your account.")
    elif "Single" in plan:
        st.session_state["purchased_credits"] += 1
        st.success("Purchased 1 credit added to your account.")
    elif "Monthly" in plan:
        st.session_state["unlimited_until"] = (datetime.now() + timedelta(days=30)).isoformat()
        st.success("Monthly unlimited activated for 30 days.")
    st.experimental_set_query_params()

st.markdown("### Customer / Site Info")
colA, colB = st.columns(2)
with colA:
    customer_name = st.text_input("Customer Name")
with colB:
    address = st.text_input("Property Address")
notes = st.text_area("Notes (optional)")

st.markdown("#### Your Company / Technician (shown on the report)")
colC, colD, colE = st.columns(3)
with colC:
    tech_name = st.text_input("Technician / Assessor Name", value="")
with colD:
    tech_license = st.text_input("License / Cert # (optional)", value="")
with colE:
    tech_phone = st.text_input("Contact Phone", value="")

st.markdown("### Upload Roof Images")
uploaded_files = st.file_uploader(
    "Upload one or more images (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
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
allow = True
if "last_ts" in st.session_state:
    elapsed = time.time() - st.session_state["last_ts"]
    if elapsed < COOLDOWN_SECONDS:
        allow = False
        st.info(f"Cooldown: wait {int(COOLDOWN_SECONDS - elapsed)}s before next run.")

if st.button("Analyze & Generate Report", type="primary", disabled=not allow):
    ok, reason = can_generate_now()
    if not ok:
        st.error(reason); st.stop()
    if not uploaded_files:
        st.error("Please upload at least one image."); st.stop()

    assessments: List[Tuple[str, RoofAssessment, Image.Image]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for idx, f in enumerate(uploaded_files, start=1):
        status.write(f"Analyzing image {idx}/{len(uploaded_files)} …")
        try:
            pil = normalize_image(f)
            img_b64 = image_to_b64(pil)
            a, raw = call_openai_vision(img_b64)
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

    # Consume a credit/free use
    consume_credit()

    # Aggregate summary
    avg_damage = sum(a.damage_percentage for _, a, __ in assessments) / len(assessments)
    max_urgency = max(a.urgency for _, a, __ in assessments)
    avg_est = sum(a.estimate_usd for _, a, __ in assessments) / len(assessments)

    st.markdown("### Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Damage %", f"{avg_damage:.1f}%")
    c2.metric("Max Urgency", max_urgency)
    c3.metric("Avg Estimate", fmt_currency_block(avg_est, currency))

    if any([customer_name, address, notes]):
        st.write("### Job Details")
        if customer_name: st.write(f"- **Customer:** {customer_name}")
        if address:       st.write(f"- **Address:** {address}")
        if notes:         st.write(f"- **Notes:** {notes}")

    # Friendlier summary (includes job + next steps)
    summary_text = (
        f"For {customer_name or 'the client'} at {address or 'the property'}, the roof shows ~{avg_damage:.0f}% average damage "
        f"with a maximum urgency of {max_urgency}/10. Estimated average repair cost is {fmt_currency_block(avg_est, currency)}. "
        "Recommended next steps: (1) tarp/temporary waterproofing if leaks are present, "
        "(2) schedule contractor inspection within 7 days, "
        "(3) gather additional photos for insurer if filing a claim."
    )

    report_id = new_report_id()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    md_report = build_markdown_report(
        report_id, ts, customer_name, address, notes, currency,
        assessments, summary_text, tech_name, tech_license, tech_phone
    )
    st.markdown("---")
    st.markdown("## Report Preview (Markdown)")
    st.markdown(md_report)

    pdf_bytes = build_pdf_report(
        report_id, ts, customer_name, address, notes, currency,
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

    st.session_state["last_ts"] = time.time()

# Footer
st.markdown("---")
cols = st.columns(3)
cols[0].markdown("[Terms of Service](#)")
cols[1].markdown("[Privacy Policy](#)")
cols[2].markdown(f"© {datetime.now().year} {COMPANY_NAME} • v{APP_VERSION}")
