print("\n--- PHASE 4: DEPLOYING LIVE WEB APP ---")
def analyze_video(video_path):
    if video_path is None: return "Please upload a video."
    vis_input = np.expand_dims(extract_face_frames(video_path), axis=0)
    aud_input = np.expand_dims(extract_mel_spectrogram(video_path), axis=0)
    prediction = multimodal_model.predict([vis_input, aud_input])[0][0]
    
    if prediction > 0.5:
        return f"🚨 SYNTHETIC MEDIA DETECTED\nConfidence: {(prediction * 100):.2f}%\nReason: Lip-speech spatial/audio mismatch flagged."
    else:
        return f"✅ AUTHENTIC MEDIA\nConfidence: {((1 - prediction) * 100):.2f}%\nReason: Spatial and audio streams align seamlessly."

app = gr.Interface(
    fn=analyze_video,            
    inputs=gr.Video(),           
    outputs=gr.Textbox(label="Detection Results"), 
    title="Multimodal Deepfake Detection Engine",
    description="Upload an .mp4 file. The dual-stream CNN analyzes spatial facial landmarks and audio Mel-spectrograms for synchronization anomalies."
)

try:
    app.launch(share=True, debug=True, inline=False)
except Exception as e:
    print(f"Gradio Launch Error: {e}")
