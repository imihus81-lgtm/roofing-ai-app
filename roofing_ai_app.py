import streamlit as st
from openai import OpenAI
import stripe
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import time  # Added for retry logic

# Load environment variables
load_dotenv()

# Configure APIs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def analyze_roof_damage(image_data):
    """Analyze roof damage using GPT-4o with retry logic"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Analyze this roof for damage. Return in this format:\n\n"
                                        "1. Damage Percentage: [number]%\n"
                                        "2. Urgency Level: [1-10]\n"
                                        "3. Repair Estimate: $[amount]\n"
                                        "4. Detailed Analysis: [2-3 sentences]\n"
                                        "5. Recommended Action: [brief advice]"
                            },
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"  # Fixed comma
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"Error analyzing image after {max_retries} attempts: {str(e)}"

def create_checkout_session(plan_name, plan_price):
    """Create a Stripe checkout session"""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f'Storm Scout AI - {plan_name}',
                    },
                    'unit_amount': plan_price * 100,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=st.query_params.get('success_url', ['http://localhost:8501'])[0],
            cancel_url=st.query_params.get('cancel_url', ['http://localhost:8501'])[0],
        )
        return session.url
    except Exception as e:
        st.error(f"Payment error: {str(e)}")
        return None

def main():
    """Main application"""
    st.set_page_config(
        page_title="Storm Scout AI - Roof Damage Analysis",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Storm Scout AI")
    st.markdown("### Professional Roof Damage Analysis & Reporting")
    
    # Sidebar with payment options
    with st.sidebar:
        st.header("💰 Pricing Plans")
        
        plan = st.radio(
            "Choose a plan:",
            ["Single Report - $49", "5 Reports - $199", "Monthly Unlimited - $299"]
        )
        
        if st.button("Subscribe Now", type="primary"):
            if "Single Report" in plan:
                url = create_checkout_session("Single Report", 49)
            elif "5 Reports" in plan:
                url = create_checkout_session("5 Reports Pack", 199)
            else:
                url = create_checkout_session("Monthly Unlimited", 299)
            
            if url:
                st.markdown(f"[Complete Payment]({url})")
            else:
                st.error("Payment system temporarily unavailable")
        
        st.markdown("---")
        st.info("""
        **What you get:**
        - Instant damage analysis
        - Insurance-ready reports
        - Repair cost estimates
        - Professional recommendations
        """)
    
    # Main content area with damage analysis
    st.header("Upload Roof Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a roof image", 
        type=["jpg", "jpeg", "png"],
        help="Upload clear photos of roof damage for best results"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            
            # Resize image to reduce size
            image = image.resize((800, 600))
            st.image(image, use_column_width=True)
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)  # Reduced quality
            img_str = base64.b64encode(buffered.getvalue()).decode()
        
        with col2:
            st.subheader("Damage Analysis")
            if st.button("Analyze Damage", type="primary"):
                with st.spinner("Analyzing damage..."):
                    analysis = analyze_roof_damage(img_str)
                
                if analysis.startswith("Error"):
                    st.error(analysis)
                else:
                    st.success("Analysis Complete!")
                    st.text_area("Damage Report", analysis, height=300)
                    
                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=analysis,
                        file_name="roof_damage_report.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
