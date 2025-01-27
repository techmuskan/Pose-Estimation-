import streamlit as st
import cv2
import numpy as np
from pose_estimator import PoseEstimator
from utils import draw_fps, apply_futuristic_overlay
import time
# from download import create_project_zip
import os

# Initialize pose estimator
pose_estimator = PoseEstimator()

# Page configuration
st.set_page_config(
    page_title="Human Pose Estimation",
    page_icon="🤖",
    layout="wide",
)

# Debugging statement
st.write("Streamlit app is running")

st.title("Human Pose Estimation")
st.write("This is the main content of the app.")

# Custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Title and description with custom HTML styling
st.markdown("""
    <div class='header'>
        <h1>🤖 Human Pose Estimation</h1>
        <p class='subtitle'>Advanced real-time pose detection powered by machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h2 style='margin-bottom: 0.5rem;'>⚙️ Controls</h2>
            <p style='color: #a8b2d1; font-size: 0.9rem;'>Configure your detection settings</p>
        </div>
    """, unsafe_allow_html=True)

    input_source = st.radio("Select Input Source", ["Webcam", "Upload Image"])

    st.markdown("---")

    # Download section
    st.markdown("""
        <div style='margin-top: 2rem; margin-bottom: 1rem;'>
            <h3>📥 Download Project</h3>
            <p stylye='color: #a8b2d1; font-size: 0.9rem;'>Get the complete source code</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("📦 Download Source Code"):
        create_project_zip()
        # Get the latest zip file
        zip_files = [f for f in os.listdir() if f.startswith("pose_estimation_project_") and f.endswith(".zip")]
        if zip_files:
            latest_zip = max(zip_files, key=os.path.getctime)  # Get the most recent zip file using creation time
            with open(latest_zip, "rb") as fp:
                btn = st.download_button(
                    label="⬇️ Click to Download",
                    data=fp,
                    file_name=latest_zip,
                    mime="application/zip"
                )

    # System status with icons
    st.markdown("""
        <div style='margin-top: 2rem;'>
            <h3>📊 System Status</h3>
            <p>🔵 Model: MediaPipe Pose</p>
            <p>⚡ Mode: Real-time</p>
            <p>🎯 Status: Active</p>
        </div>
    """, unsafe_allow_html=True)

# Main content with improved layout
col1, col2 = st.columns([2, 1])

with col1:
    if input_source == "Webcam":
        st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h3>📹 Live Feed</h3>
                <p style='color: #a8b2d1;'>Real-time pose detection from your camera</p>
            </div>
        """, unsafe_allow_html=True)

        video_placeholder = st.empty()

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Camera not accessible. Please check your connection or try image upload.")
            else:
                col3, col4 = st.columns(2)
                with col3:
                    start_button = st.button("▶️ Start Detection")
                with col4:
                    stop_button = st.button("⏹️ Stop")

                if start_button:
                    st.markdown("""
                        <div class='info-box' style='padding: 1rem; border-radius: 8px; background: rgba(0, 180, 216, 0.1); margin: 1rem 0;'>
                            <p>💡 Tip: Stand in clear view of the camera for best results</p>
                        </div>
                    """, unsafe_allow_html=True)

                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from webcam")
                            break

                        # Process frame
                        start_time = time.time()
                        results = pose_estimator.process_frame(frame)
                        frame = pose_estimator.draw_landmarks(frame, results)

                        # Calculate and draw FPS
                        fps = 1 / (time.time() - start_time)
                        frame = draw_fps(frame, fps)

                        # Apply futuristic overlay
                        frame = apply_futuristic_overlay(frame)

                        # Display frame
                        video_placeholder.image(frame, channels="BGR", use_column_width=True)

                    cap.release()
        except Exception as e:
            st.error(f"❌ Camera Error: {str(e)}")

    else:
        st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h3>📤 Image Upload</h3>
                <p style='color: #a8b2d1;'>Analyze poses in your photos</p>
            </div>
        """, unsafe_allow_html=True)

        # File upload area with instructions
        st.markdown("""
            <div style='padding: 1.5rem; background: rgba(23, 42, 69, 0.7); border-radius: 12px; margin-bottom: 1rem;'>
                <h4 style='color: #00b4d8; margin-bottom: 0.5rem;'>📋 Upload Guidelines</h4>
                <p>• Supported formats: JPG, JPEG, PNG</p>
                <p>• Clear, well-lit images work best</p>
                <p>• Ensure the full body is visible</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)

                # Process image
                results = pose_estimator.process_frame(frame)
                frame = pose_estimator.draw_landmarks(frame, results)
                frame = apply_futuristic_overlay(frame)

                st.image(frame, channels="BGR", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

# Analytics panel with improved visuals
with col2:
    st.markdown("""
        <div class='analytics-panel'>
            <h2>🔍 Pose Analytics</h2>
            <p style='color: #a8b2d1; margin-bottom: 1rem;'>Real-time pose measurements and analysis</p>
        </div>
    """, unsafe_allow_html=True)

    if 'results' in locals() and results and results.pose_landmarks:
        landmarks = pose_estimator.get_landmark_coordinates(results)
        if landmarks:
            # Confidence score with improved visualization
            st.markdown("### 📊 Confidence Score")
            confidence = pose_estimator.get_pose_confidence(results)
            col5, col6 = st.columns([3, 1])
            with col5:
                st.progress(confidence)
            with col6:
                st.markdown(f"<h4 style='color: #00b4d8; margin: 0;'>{confidence:.0%}</h4>", unsafe_allow_html=True)

            # Joint angles with visual indicators
            st.markdown("### 📐 Key Joint Angles")
            angles = pose_estimator.calculate_joint_angles(landmarks)
            for joint, angle in angles.items():
                st.markdown(f"""
                    <div style='background: rgba(23, 42, 69, 0.7); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                        <p style='margin: 0;'><strong>{joint}:</strong> {angle:.1f}°</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👋 No pose detected. Please ensure a person is visible in the frame.")