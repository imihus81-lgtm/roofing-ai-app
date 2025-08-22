# 🏠 Storm Scout AI - Roof Damage Analysis

AI-powered roof damage analysis tool that provides instant insurance-ready reports with repair estimates.

## ✨ Features

- **AI-Powered Analysis**: GPT-4 Vision for accurate damage detection
- **Instant Reports**: Generate professional reports in seconds
- **Payment Integration**: Stripe integration for seamless payments
- **24/7 Availability**: Always available for storm damage assessment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API account
- Stripe account

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/roofing-ai-app.git
   cd roofing-ai-app
   # right after: md = render_markdown_report(assessment)
# add a thumbnail (data URI) to the bottom of the report

def _img_to_base64_md(img):
    import io, base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"![Analyzed Roof](data:image/jpeg;base64,{b64})"

md = render_markdown_report(assessment)
md += "\n\n### Photo\n" + _img_to_base64_md(img)
