import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.tracker import process_video, analytics_from_rows
from src.utils import normalize_heatmap

st.set_page_config(page_title="CV Smart Tracker", page_icon="🎥", layout="wide")

st.markdown("""
<style>
/* modern dark-ish but readable */
.main { background: #0b1220; }
.block-container { padding-top: 1.2rem; }
h1, h2, h3, p, label, span { color: #e8eefc !important; }
.stButton button { border-radius: 14px; padding: 0.6rem 1rem; font-weight: 700; }
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
}
.small { opacity: 0.85; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎥 YOLOv8 Smart Tracker + Analytics Dashboard")
st.caption("Upload a video → detect + track → heatmap + CSV analytics. (Looks insane on GitHub/LinkedIn.)")

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.01)
    max_frames = st.slider("Max frames to process (demo mode)", 50, 1200, 300, 50)
    st.markdown("---")
    st.markdown("### ✅ Output")
    show_heatmap = st.checkbox("Show heatmap", True)
    show_tables = st.checkbox("Show analytics tables", True)

uploaded = st.file_uploader("📤 Upload an MP4 video", type=["mp4", "mov", "m4v"])
if not uploaded:
    st.info("Upload a video to start.")
    st.stop()

# Save upload
tmp_path = "assets/input_video.mp4"
with open(tmp_path, "wb") as f:
    f.write(uploaded.read())

colA, colB = st.columns([1.4, 1.0], gap="large")
with colA:
    st.markdown('<div class="card"><h3>🧠 Live Tracking Preview</h3><p class="small">Processing frames…</p></div>', unsafe_allow_html=True)
    frame_slot = st.empty()

with colB:
    st.markdown('<div class="card"><h3>📊 Live Stats</h3><p class="small">Counts update while frames process.</p></div>', unsafe_allow_html=True)
    stat_slot = st.empty()

heat_final = None
rows_final = []

# Run processing
progress = st.progress(0)
total = max_frames

for idx, (annotated, heat, rows) in enumerate(process_video(tmp_path, conf=conf, max_frames=max_frames)):
    rows_final = rows
    heat_final = heat

    # display frame
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_slot.image(rgb, use_column_width=True)  # <-- avoids your use_container_width error

    # stats
    if rows_final:
        df_live = pd.DataFrame(rows_final)
        top = df_live["class"].value_counts().head(6)
        stat_slot.write(top)
    else:
        stat_slot.write("No detections yet…")

    progress.progress(min((idx + 1) / total, 1.0))

st.success("✅ Done processing!")

df, per_class, tracks = analytics_from_rows(rows_final)

st.markdown("### 📌 Results")
c1, c2, c3 = st.columns(3)
c1.metric("Total detections", int(len(df)) if not df.empty else 0)
c2.metric("Classes found", int(df["class"].nunique()) if not df.empty else 0)
c3.metric("Unique tracked objects", int(tracks["unique_objects"].sum()) if not tracks.empty else 0)

if show_heatmap and heat_final is not None:
    st.markdown("### 🔥 Movement Heatmap")
    hm = normalize_heatmap(heat_final)
    hm_img = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_TURBO)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    st.image(hm_color, caption="Where objects appeared the most", use_column_width=True)

if show_tables:
    st.markdown("### 📄 Analytics Tables")
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Detections per class")
        st.dataframe(per_class, use_container_width=True)
    with t2:
        st.subheader("Unique objects per class (tracked IDs)")
        st.dataframe(tracks, use_container_width=True)

st.markdown("### ⬇️ Export CSV")
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download detections.csv", data=csv, file_name="detections.csv", mime="text/csv")
else:
    st.info("No detections to export.")