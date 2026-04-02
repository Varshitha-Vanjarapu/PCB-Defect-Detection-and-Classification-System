import streamlit as st
from PIL import Image
import io
import os
import pandas as pd
from inference import run_pcb_inference 

st.set_page_config(page_title="PCB Defects Detector", layout="wide")

# --- ADVANCED CSS INJECTION ---
st.markdown("""
    <style>
    /* Deep gradient background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    /* Glowing Gradient Title */
    .glow-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }

    /* Vibrant Gradient Badges with Hover Animation */
    .badge-base {
        border-radius: 20px;
        padding: 10px 15px;
        text-align: center;
        font-weight: bold;
        color: white;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .badge-base:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255,255,255,0.2);
    }
    
    /* Specific Vivid Colors for each defect */
    .b-hole { background: linear-gradient(45deg, #ff0844, #ffb199); }
    .b-bite { background: linear-gradient(45deg, #f83600, #f9d423); }
    .b-open { background: linear-gradient(45deg, #00c6ff, #0072ff); }
    .b-short { background: linear-gradient(45deg, #11998e, #38ef7d); }
    .b-spur { background: linear-gradient(45deg, #b224ef, #7579ff); }
    .b-copper { background: linear-gradient(45deg, #f12711, #f5af19); }

    /* Make standard markdown text white for contrast */
    .stMarkdown, p {
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Use the new glowing title class
st.markdown('<div class="glow-title">PCB Defects Detector</div>', unsafe_allow_html=True)

# Apply the new vibrant badge classes
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.markdown('<div class="badge-base b-hole">Missing Hole</div>', unsafe_allow_html=True)
col2.markdown('<div class="badge-base b-bite">Mouse Bite</div>', unsafe_allow_html=True)
col3.markdown('<div class="badge-base b-open">Open Circuit</div>', unsafe_allow_html=True)
col4.markdown('<div class="badge-base b-short">Short Circuit</div>', unsafe_allow_html=True)
col5.markdown('<div class="badge-base b-spur">Spur</div>', unsafe_allow_html=True)
col6.markdown('<div class="badge-base b-copper">Spurious Copper</div>', unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("Upload a PCB Image to instantly scan for defects...", type=["jpg", "jpeg", "png"])

colA, colB = st.columns(2)
with colA:
    st.markdown("**ORIGINAL PCB**")
    img_placeholder = st.empty()
with colB:
    st.markdown("**DETECTION OVERLAY**")
    overlay_placeholder = st.empty()

results_text = st.empty()

if uploaded_file is None:
    results_text.write("Awaiting image upload...")
else:
    image = Image.open(uploaded_file).convert('RGB')
    img_placeholder.image(image, use_container_width=True)
    results_text.write("⏳ Analyzing board...") 
    
    if not os.path.exists("best.pt"):
        st.error("🚨 YOLO weights not found! Ensure 'best.pt' is in the same folder.")
        st.stop()
        
    annotated_img, num_defects, found_classes, df = run_pcb_inference(image, "best.pt")
    
    overlay_placeholder.image(annotated_img, use_container_width=True)
    results_text.empty() 
    
    if num_defects > 0:
        st.markdown(f"**Analysis Complete!**")
        st.markdown(f"**Defects Found:** {num_defects}")
        st.markdown(f"**Types:** {', '.join(found_classes)}")
        
        st.markdown("---")
        
        st.markdown("### 📊 Prediction Log")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### 📥 Export Results")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        img_pil = Image.fromarray(annotated_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(label="📄 Download Prediction Log (CSV)", data=csv_data, file_name='pcb_defect_log.csv', mime='text/csv', use_container_width=True)
        with dl_col2:
            st.download_button(label="🖼️ Download Annotated Image", data=byte_im, file_name="annotated_pcb.jpg", mime="image/jpeg", use_container_width=True)
    else:
        st.markdown(f"**Analysis Complete!**")
        st.markdown(f"**Defects Found:** 0")
        st.success("No defects detected. Board passes quality check.")