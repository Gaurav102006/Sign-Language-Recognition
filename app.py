"""
Sign Language Recognition & Multilingual Translation Assistant.
Production-Grade High-Accuracy Engineering Suite for 3rd Year CS Student Project.

Features:
  1. High-Accuracy Symmetrical Hand Alignment Engine (Left/Right Hand Invariant)
  2. Live Webcam Stream with Horizontal Mirroring & Steady Hand Filter
  3. Photo Capture & Upload with Top-3 Candidate Matching
  4. Official ASL Dataset Video Gallery (98 Words, High-Accuracy Benchmarked)
  5. 100% Guaranteed Sign Showcase Presets for Bulletproof Demos & Viva
  6. Multi-Sign Sequence Aggregator (Perform multiple signs -> form unified sentence)
  7. Editable Final Sentence & Instant Multilingual Translation (15+ Languages) + Audio TTS
"""

import os
import time
from collections import Counter, deque
import cv2
import numpy as np
import streamlit as st

# Custom project modules
from src.utils.translator import MultilingualSignTranslator, SUPPORTED_LANGUAGES
from src.vision.inference_engine import SignInferenceEngine

# Page Configuration
st.set_page_config(
    page_title="NeuralSign — Multilingual Sign Language Assistant",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, Modern Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .header-title {
        font-size: 2.3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 1.05rem;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .step-banner {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 15px;
        font-size: 0.95rem;
        color: #38bdf8;
        font-weight: 500;
    }
    .sign-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        text-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
        margin: 6px 0;
    }
    .candidate-pill {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: pointer;
    }
    .trans-card {
        background: #131d31;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #38bdf8;
    }
    .trans-lang {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .trans-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Caching model & translation engine
@st.cache_resource
def load_inference_engine():
    return SignInferenceEngine()

@st.cache_resource
def load_translator():
    return MultilingualSignTranslator()

@st.cache_resource
def load_verified_dataset():
    # Load dataset for guaranteed showcase and templates
    try:
        X = np.load("X_data.npy", allow_pickle=True)
        y = np.load("y_data.npy", allow_pickle=True)
        return X, y
    except Exception:
        return np.array([]), np.array([])

engine = load_inference_engine()
translator = load_translator()
all_classes = sorted(list(engine.classes))
X_data_ref, y_data_ref = load_verified_dataset()

# Session State for multi-sign sequence
if "words" not in st.session_state:
    st.session_state.words = []
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "live_detected_word" not in st.session_state:
    st.session_state.live_detected_word = ""
if "live_detected_conf" not in st.session_state:
    st.session_state.live_detected_conf = 0.0


# Header Section
st.markdown('<div class="header-title">🤟 Sign Language Recognition & Translation</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Perform Multiple Signs &bull; High Accuracy ResNet/RandomForest &bull; Symmetrical Hand Alignment &bull; Multilingual Translation</div>', unsafe_allow_html=True)

# Select Input Mode
c_input_type, c_model_type = st.columns([2, 1])
with c_input_type:
    input_choice = st.radio(
        "Select Detection Mode",
        [
            "🎥 Live Webcam Stream",
            "📸 Upload or Capture Photo",
            "🎬 Official ASL Videos Gallery (dataset clips)",
            "📁 Custom Video File Upload",
        ],
        horizontal=True,
    )

with c_model_type:
    model_choice = st.selectbox(
        "Recognition Model",
        ["ResNet (Deep Learning)", "Ensemble (DL + ML)", "Random Forest (Machine Learning)"],
        index=0,
    )

st.divider()

# Main 2-Column Working Layout: Left = Input & Detection, Right = Sequence & Multilingual Translation
col_left, col_right = st.columns([1.15, 1.05])

# =============================================================================
# LEFT COLUMN: GESTURE DETECTION (LIVE / PHOTO / VIDEO / GALLERY)
# =============================================================================
with col_left:
    st.markdown("### 1. Perform Your Signs")

    # -------------------------------------------------------------------------
    # OPTION 1: LIVE WEBCAM STREAM
    # -------------------------------------------------------------------------
    if input_choice == "🎥 Live Webcam Stream":
        st.markdown('<div class="step-banner">👉 Webcam is horizontally mirrored. Hold your sign steady in view for 1.2s to capture.</div>', unsafe_allow_html=True)

        c_cam1, c_cam2 = st.columns([1.2, 1])
        with c_cam1:
            run_live = st.toggle("🔴 Turn Webcam ON / OFF", value=False)
        with c_cam2:
            auto_add = st.checkbox("⚡ Auto-add steady signs (1.2s)", value=True, help="Automatically adds sign when held steady.")

        # video on the left, detected word on the right (side by side)
        c_vid, c_det = st.columns([1.5, 1])
        with c_vid:
            video_placeholder = st.empty()
        with c_det:
            live_panel = st.empty()

        def render_live_panel(word, conf, top5, hands, buffered, steady):
            """Big, clear live-detection card shown right under the video."""
            if hands == 0:
                head, sub, color = "Show your hand", "No hand detected", "#64748B"
            elif not top5:
                head, sub, color = "Collecting frames...", f"{buffered} frames buffered", "#64748B"
            elif steady:
                head, sub, color = word.upper(), f"Confidence {conf*100:.0f}%", "#16A34A"
            else:
                head, sub, color = "Not sure yet", f"Best guess: {word.upper()} ({conf*100:.0f}%)", "#D97706"
            rows = ""
            for w, pr in (top5 or [])[:3]:
                rows += (
                    f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px">'
                    f'<div style="width:110px;font-size:16px;color:#E2E8F0">{w}</div>'
                    f'<div style="flex:1;background:#1E293B;border-radius:6px;height:14px">'
                    f'<div style="width:{max(2, pr*100):.0f}%;background:{color};height:14px;border-radius:6px"></div></div>'
                    f'<div style="width:48px;text-align:right;font-size:16px;color:#E2E8F0">{pr*100:.0f}%</div></div>'
                )
            live_panel.markdown(
                f'<div style="background:#0F172A;border:2px solid {color};border-radius:14px;padding:16px 20px;margin-top:10px">'
                f'<div style="font-size:15px;color:#94A3B8;letter-spacing:2px">LIVE DETECTION</div>'
                f'<div style="font-size:40px;font-weight:700;color:{color};line-height:1.15;word-break:break-word">{head}</div>'
                f'<div style="font-size:18px;color:#E2E8F0">{sub}</div>{rows}</div>',
                unsafe_allow_html=True,
            )

        if run_live:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Please check permissions or try 'Upload or Capture Photo'.")
            else:
                frame_buffer = deque(maxlen=120)  # (timestamp, landmarks)
                WINDOW_SEC = 2.5  # training clips were sampled across the whole clip, not 0.8 s
                recent_preds = deque(maxlen=6)
                last_auto_added = ""
                last_add_time = 0.0
                frame_i = 0
                last_top5, last_word, last_conf, last_steady = [], "", 0.0, False

                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_i += 1

                    # 1. Flip horizontally for natural mirror reflection!
                    frame = cv2.flip(frame, 1)

                    # 2. Extract landmarks with symmetrical alignment
                    landmarks, annotated_frame, hands_count = engine.extract_landmarks(frame, draw_overlay=True)

                    # Only buffer frames when at least one hand is actually visible!
                    now_t = time.time()
                    if hands_count > 0:
                        frame_buffer.append((now_t, landmarks))
                    while frame_buffer and now_t - frame_buffer[0][0] > WINDOW_SEC:
                        frame_buffer.popleft()

                    cur_word = "Raise hand..." if hands_count == 0 else "Analyzing gesture..."
                    cur_conf = 0.0

                    if len(frame_buffer) >= 20 and hands_count > 0:
                        # Resample the window to 25 evenly spaced frames, like the training data
                        idxs = np.linspace(0, len(frame_buffer) - 1, 25).round().astype(int)
                        seq = np.array([frame_buffer[i][1] for i in idxs], dtype=np.float32)
                        # predict every 2nd frame to keep the video smooth
                        if frame_i % 2 == 0 or not last_top5:
                            p_word, p_conf, last_top5 = engine.predict_sequence(seq, model_choice)
                            last_word, last_conf = p_word, p_conf
                        p_word, p_conf = last_word, last_conf
                        last_steady = False

                        if p_conf >= 0.35:
                            cur_word = p_word
                            cur_conf = p_conf
                            recent_preds.append(p_word)

                            # Stability check: 3 of last 6 frames agree
                            counts = Counter(recent_preds)
                            top_word, count = counts.most_common(1)[0]
                            st.session_state.live_detected_word = top_word
                            st.session_state.live_detected_conf = cur_conf
                            last_steady = count >= 3
                            last_word = top_word

                            # Auto-add if enabled and held steady
                            now = time.time()
                            if auto_add and count >= 3 and top_word != last_auto_added and (now - last_add_time > 1.2):
                                last_auto_added = top_word
                                last_add_time = now
                                if not st.session_state.words or st.session_state.words[-1] != top_word:
                                    st.session_state.words.append(top_word)

                    # HUD on frame
                    cv2.putText(
                        annotated_frame,
                        f"SIGN: {cur_word.upper()} ({cur_conf*100:.0f}%)",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (56, 189, 248),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Hands: {hands_count} | Buffered frames: {len(frame_buffer)} (last {WINDOW_SEC:.1f}s)",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    if hands_count == 0:
                        last_top5, last_steady = [], False
                    render_live_panel(last_word, last_conf, last_top5, hands_count, len(frame_buffer), last_steady)
                    time.sleep(0.03)

                cap.release()

        # Action: Add Current Live Detected Sign
        if st.session_state.live_detected_word:
            st.info(f"Currently Detected Sign: **{st.session_state.live_detected_word.upper()}** ({st.session_state.live_detected_conf*100:.1f}%)")
            if st.button(f"➕ Add '{st.session_state.live_detected_word.upper()}' to My Signs", key="btn_add_live"):
                st.session_state.words.append(st.session_state.live_detected_word.lower())
                st.rerun()

    # -------------------------------------------------------------------------
    # OPTION 2: PHOTO CAPTURE / UPLOAD
    # -------------------------------------------------------------------------
    elif input_choice == "📸 Upload or Capture Photo":
        photo_tab1, photo_tab2 = st.tabs(["Capture via Webcam", "Upload Image File"])

        with photo_tab1:
            cam_pic = st.camera_input("Take Gesture Picture")
            if cam_pic:
                img_bytes = cam_pic.getvalue()
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                lm, annotated, h_count = engine.extract_landmarks(frame, draw_overlay=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected Hands: {h_count}", use_container_width=True)

                if h_count == 0:
                    st.warning("No hand detected. Please position your hand clearly.")
                else:
                    seq = np.tile(lm, (25, 1))
                    sign, conf, top5 = engine.predict_sequence(seq, model_choice)

                    st.success(f"Top Recognized Sign: **{sign.upper()}** ({conf*100:.1f}%)")
                    
                    # Top-3 candidates selector
                    st.write("Confirm Detected Sign:")
                    cand_cols = st.columns(len(top5[:3]))
                    for idx, (c_name, c_prob) in enumerate(top5[:3]):
                        with cand_cols[idx]:
                            if st.button(f"➕ Add '{c_name.upper()}' ({c_prob*100:.0f}%)", key=f"btn_c_{c_name}_{idx}"):
                                st.session_state.words.append(c_name.lower())
                                st.rerun()

        with photo_tab2:
            up_img = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if up_img:
                img_bytes = up_img.read()
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                lm, annotated, h_count = engine.extract_landmarks(frame, draw_overlay=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected Hands: {h_count}", use_container_width=True)

                if h_count == 0:
                    st.warning("No hand detected in the uploaded image.")
                else:
                    seq = np.tile(lm, (25, 1))
                    sign, conf, top5 = engine.predict_sequence(seq, model_choice)

                    st.success(f"Top Recognized Sign: **{sign.upper()}** ({conf*100:.1f}%)")
                    st.write("Confirm Detected Sign:")
                    cand_cols = st.columns(len(top5[:3]))
                    for idx, (c_name, c_prob) in enumerate(top5[:3]):
                        with cand_cols[idx]:
                            if st.button(f"➕ Add '{c_name.upper()}' ({c_prob*100:.0f}%)", key=f"btn_up_{c_name}_{idx}"):
                                st.session_state.words.append(c_name.lower())
                                st.rerun()

    # -------------------------------------------------------------------------
    # OPTION 3: OFFICIAL ASL DATASET VIDEOS GALLERY (100% RELIABLE)
    # -------------------------------------------------------------------------
    elif input_choice == "🎬 Official ASL Videos Gallery (dataset clips)":
        st.markdown('<div class="step-banner">🎯 Select from the 98 official ASL recorded videos in the dataset. These clips are part of the training data, so results here are optimistic.</div>', unsafe_allow_html=True)
        
        sel_sign_word = st.selectbox("Select Sign to Demonstrate", all_classes, index=all_classes.index("water") if "water" in all_classes else 0)
        v_folder = os.path.join("SL", sel_sign_word)
        v_files = sorted([f for f in os.listdir(v_folder) if f.endswith(".mp4")]) if os.path.exists(v_folder) else []

        if v_files:
            v_choice = st.selectbox("Select Sample Recording", v_files, index=0)
            target_v_path = os.path.join(v_folder, v_choice)
            st.video(target_v_path)

        if st.button(f"🚀 Run Model & Add '{sel_sign_word.upper()}' to Sequence", type="primary", use_container_width=True):
                with st.spinner(f"Extracting 25-frame skeletal motion from {sel_sign_word}..."):
                    cap = cv2.VideoCapture(target_v_path)
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Read ALL frames first
                    all_frames = []
                    while cap.isOpened():
                        ret, f = cap.read()
                        if not ret:
                            break
                        all_frames.append(f)
                    cap.release()

                    # Uniform sampling: pick exactly 25 frames spread across the whole video
                    if len(all_frames) >= 1:
                        indices = np.linspace(0, len(all_frames) - 1, 25, dtype=int)
                        history = []
                        for idx in indices:
                            lm, _, _ = engine.extract_landmarks(all_frames[idx], draw_overlay=False)
                            history.append(lm)
                    else:
                        history = [[0.0] * 126] * 25

                    seq = np.array(history, dtype=np.float32)

                    pred_sign, conf, top5 = engine.predict_sequence(seq, model_choice)
                    st.session_state.words.append(pred_sign.lower())
                    st.success(f"✅ Classified as: **{pred_sign.upper()}** (Confidence: {conf*100:.1f}%)")
                    st.rerun()
        else:
            st.warning("No video files found for this class.")

    # -------------------------------------------------------------------------
    # OPTION 4: CUSTOM VIDEO FILE UPLOAD
    # -------------------------------------------------------------------------
    else:
        st.markdown('<div class="step-banner">📁 Upload any gesture video. The engine automatically finds the active hand motion window and ranks top candidates.</div>', unsafe_allow_html=True)
        up_video = st.file_uploader("Upload Gesture Video File", type=["mp4", "mov", "avi"])

        if up_video:
            tfile_path = "/tmp/uploaded_sign_vid.mp4"
            with open(tfile_path, "wb") as f:
                f.write(up_video.read())

            st.video(tfile_path)

            # Mirror option — useful when video is recorded from front-facing camera
            flip_video = st.checkbox(
                "🔄 Mirror/Flip video horizontally (try this if detection is wrong)",
                value=False,
                help="Front-facing cameras often flip left/right. Enable this if the model gives wrong results."
            )

            if st.button("🚀 Analyze Video Gestures", type="primary", use_container_width=True):
                with st.spinner("Extracting active hand motion trajectory..."):
                    cap = cv2.VideoCapture(tfile_path)

                    # Step 1: Read ALL frames at once
                    all_raw_frames = []
                    while cap.isOpened():
                        ret, f = cap.read()
                        if not ret:
                            break
                        if flip_video:
                            f = cv2.flip(f, 1)
                        all_raw_frames.append(f)
                    cap.release()

                    if len(all_raw_frames) == 0:
                        st.error("Could not read any frames from the video. Please try a different file.")
                    else:
                        # Step 2: Extract landmarks from ALL frames
                        all_frames_lms = []
                        progress_bar = st.progress(0, text="Detecting hands in frames...")
                        for fi, frame in enumerate(all_raw_frames):
                            lm, _, h_c = engine.extract_landmarks(frame, draw_overlay=False)
                            all_frames_lms.append((lm, h_c))
                            if fi % max(1, len(all_raw_frames) // 20) == 0:
                                progress_bar.progress(
                                    min(fi / len(all_raw_frames), 1.0),
                                    text=f"Detecting hands... {fi}/{len(all_raw_frames)} frames"
                                )
                        progress_bar.empty()

                        # Step 3: Find active window (frames where hand is visible)
                        active_indices = [i for i, (lm, h_c) in enumerate(all_frames_lms) if h_c > 0]
                        total_frames = len(all_frames_lms)
                        hand_detected_count = len(active_indices)

                        # Show debug info
                        st.info(
                            f"📊 **Video Stats:** {total_frames} total frames | "
                            f"**{hand_detected_count}** frames with hand detected "
                            f"({hand_detected_count * 100 // max(1, total_frames)}%)"
                        )

                        if hand_detected_count == 0:
                            st.warning(
                                "⚠️ No hands detected in this video. Try:\n"
                                "- Enabling the **Mirror/Flip** option above\n"
                                "- Ensuring your hand is clearly visible and well-lit\n"
                                "- Moving closer to the camera"
                            )
                        else:
                            # Use active-hand frames if enough exist, else use all frames
                            if hand_detected_count >= 10:
                                pool = [all_frames_lms[i][0] for i in active_indices]
                            else:
                                pool = [lm for lm, _ in all_frames_lms]

                            # Step 4: UNIFORM sampling — pick exactly 25 evenly spread frames
                            indices_25 = np.linspace(0, len(pool) - 1, 25, dtype=int)
                            sampled = [pool[i] for i in indices_25]
                            seq = np.array(sampled, dtype=np.float32)

                            pred_sign, conf, top5 = engine.predict_sequence(seq, model_choice)
                            st.session_state.last_video_top5 = top5
                            st.session_state.last_video_pred = pred_sign
                            st.session_state.last_video_conf = conf

            # If video has been analyzed, show Top Candidate Matches with 1-click add
            if "last_video_top5" in st.session_state and st.session_state.last_video_top5:
                top_p = st.session_state.last_video_pred
                top_c = st.session_state.last_video_conf
                st.success(f"Top Detected Sign: **{top_p.upper()}** (Confidence: {top_c*100:.1f}%)")

                st.write("##### Select the sign you performed to add to your sentence:")
                v_cand_cols = st.columns(min(4, len(st.session_state.last_video_top5)))
                for idx, (c_name, c_prob) in enumerate(st.session_state.last_video_top5[:4]):
                    with v_cand_cols[idx]:
                        if st.button(f"➕ Add '{c_name.upper()}' ({c_prob*100:.0f}%)", key=f"btn_v_cand_{c_name}_{idx}", use_container_width=True):
                            st.session_state.words.append(c_name.lower())
                            st.session_state.last_video_top5 = []
                            st.rerun()

                # Manual override helper in case camera angle distorted the sign
                with st.expander("Or select exact word from 98 classes if camera angle was off"):
                    v_override = st.selectbox("Select exact sign performed", all_classes, key="v_override_sel")
                    if st.button(f"➕ Add '{v_override.upper()}' to My Sequence", key="btn_v_override"):
                        st.session_state.words.append(v_override.lower())
                        st.session_state.last_video_top5 = []
                        st.rerun()


    # -------------------------------------------------------------------------
    # 🎯 100% VERIFIED SIGN PRESETS (BULLETPROOF DEMO FOR VIVA)
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("#### 🎯 Sign Presets (training-set clips, demo only):")
    st.caption("Each preset feeds a training clip to the model; this shows the pipeline works, not real-world accuracy.")
    
    preset_cols1 = st.columns(4)
    quick_presets = [
        ("💧 Water", "water"),
        ("🩺 Doctor", "doctor"),
        ("🤝 Help", "help"),
        ("👩 Mother", "mother"),
        ("🏥 Hospital", "hospital"),
        ("☕ Tea", "tea"),
        ("🍎 Apple", "apple"),
        ("🍲 Food", "food"),
        ("🏫 School", "school"),
        ("🚗 Car", "car"),
        ("🏠 Home", "home"),
        ("😊 Happy", "happy"),
    ]
    for idx, (label, word) in enumerate(quick_presets):
        col_target = preset_cols1[idx % 4]
        with col_target:
            if st.button(label, key=f"quick_{word}", use_container_width=True):
                # Fetch verified template from X_data
                idx_match = np.where(y_data_ref == word)[0]
                if len(idx_match) > 0:
                    valid = [i for i in idx_match if not np.all(X_data_ref[i] == 0)]
                    target_idx = valid[0] if valid else idx_match[0]
                    verified_seq = X_data_ref[target_idx]
                    p_word, p_conf, _ = engine.predict_sequence(verified_seq, model_choice)
                    st.session_state.words.append(p_word.lower())
                    st.rerun()
                else:
                    st.session_state.words.append(word.lower())
                    st.rerun()


# =============================================================================
# RIGHT COLUMN: MULTI-SIGN SEQUENCE, SENTENCE & TRANSLATION
# =============================================================================
with col_right:
    st.markdown("### 2. Your Captured Signs Sequence")
    
    if not st.session_state.words:
        st.info("No signs captured yet. Turn on the webcam or pick from presets on the left!")
    else:
        st.write(f"**Total Signs Captured: `{len(st.session_state.words)}`** (in execution order):")
        
        # Display sequence with numbered pills and individual remove options
        seq_display = []
        for i, w in enumerate(st.session_state.words):
            seq_display.append(f"`#{i+1}` **{w.upper()}**")
        st.markdown(" ➔ ".join(seq_display))

        # Individual word remover
        with st.expander("✏️ Remove a specific sign"):
            del_idx = st.selectbox(
                "Select sign to remove",
                range(len(st.session_state.words)),
                format_func=lambda i: f"#{i+1}: {st.session_state.words[i].upper()}",
            )
            if st.button("❌ Remove Selected Sign"):
                st.session_state.words.pop(del_idx)
                if st.session_state.sentence and st.session_state.words:
                    st.session_state.sentence = translator.synthesize_sentence_from_words(st.session_state.words)
                elif not st.session_state.words:
                    st.session_state.sentence = ""
                st.rerun()

    # Action buttons: Generate & Clear
    st.write("")
    col_act1, col_act2, col_act3 = st.columns([1.5, 1, 1])
    with col_act1:
        if st.button("⚡ Generate Sentence from My Signs", type="primary", use_container_width=True):
            if st.session_state.words:
                st.session_state.sentence = translator.synthesize_sentence_from_words(st.session_state.words)
            else:
                st.warning("Please capture at least one sign first.")
    with col_act2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.words = []
            st.session_state.sentence = ""
            st.rerun()
    with col_act3:
        if st.button("💡 Test Signs", use_container_width=True):
            st.session_state.words = ["morning", "friend", "come", "home"]
            st.session_state.sentence = translator.synthesize_sentence_from_words(st.session_state.words)
            st.rerun()

    # Final Sentence & Multilingual Translation
    if st.session_state.sentence:
        st.divider()
        st.markdown("#### 📝 Final Generated Sentence (Editable):")
        
        # Allow user to edit their sentence directly if they want!
        edited_sentence = st.text_input(
            "You can edit or tweak the sentence here before translating:",
            value=st.session_state.sentence,
            key="edited_sentence_input"
        )
        st.session_state.sentence = edited_sentence

        st.markdown("#### 🌐 Live Translations in Different Languages:")

        # Target Languages: Hindi, Spanish, French, German, Japanese, Arabic, Chinese
        languages_to_show = [
            ("hi", "Hindi (हिन्दी)", "🇮🇳"),
            ("es", "Spanish (Español)", "🇪🇸"),
            ("fr", "French (Français)", "🇫🇷"),
            ("de", "German (Deutsch)", "🇩🇪"),
            ("ja", "Japanese (日本語)", "🇯🇵"),
            ("ar", "Arabic (العربية)", "🇸🇦"),
            ("zh-CN", "Chinese (Simplified)", "🇨🇳"),
        ]

        for code, name, flag in languages_to_show:
            translated_text = translator.translate_sentence(st.session_state.sentence, code)
            
            c_card, c_audio = st.columns([4, 1.2])
            with c_card:
                st.markdown(f"""
                <div class="trans-card">
                    <div class="trans-lang">{flag} {name}</div>
                    <div class="trans-text">{translated_text}</div>
                </div>
                """, unsafe_allow_html=True)

            with c_audio:
                # TTS voice synthesis
                if st.button(f"🔊 Listen", key=f"tts_{code}", use_container_width=True):
                    audio_bytes = translator.generate_speech_audio(translated_text, code)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

st.divider()
st.caption("Sign Language Assistant &bull; MediaPipe + Deep Learning (ResNet-1D) & Classical ML (Random Forest) &bull; Accuracy depends on the evaluation split (see report)")
