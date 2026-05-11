# ==========================================
# 1. THE EXTRACTORS (DATA PIPELINE)
# ==========================================
def extract_face_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // max(1, num_frames))
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
        else:
            h, w, _ = frame.shape
            face_crop = frame[h//4:3*h//4, w//4:3*w//4] 
        face_resized = cv2.resize(face_crop, (224, 224))
        frames.append(face_resized / 255.0)
    cap.release()
    return np.mean(frames, axis=0) if frames else np.zeros((224, 224, 3))

def extract_mel_spectrogram(video_path):
    temp_audio_path = f"temp_audio_{np.random.randint(1000)}.wav"
    try:
        video = VideoFileClip(video_path)
        if video.audio is None: return np.zeros((224, 224, 3)) 
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        y, sr = librosa.load(temp_audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spec_resized = cv2.resize(mel_spec_db, (224, 224))
        spec_3d = np.stack((spec_resized,)*3, axis=-1)
        spec_normalized = (spec_3d - spec_3d.min()) / (spec_3d.max() - spec_3d.min() + 1e-8)
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        return spec_normalized
    except Exception:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        return np.zeros((224, 224, 3))

def prepare_baseline_data(real_dir, fake_dir, limit_per_class=100):
    visual_data, audio_data, labels = [], [], []
    
    print(f"Extracting up to {limit_per_class} Real Videos...")
    real_videos = glob.glob(os.path.join(real_dir, '**/*.mp4'), recursive=True)[:limit_per_class]
    for vid in real_videos:
        visual_data.append(extract_face_frames(vid))
        audio_data.append(extract_mel_spectrogram(vid))
        labels.append(0) # 0 = Real
        
    print(f"Extracting up to {limit_per_class} Fake Videos...")
    fake_videos = glob.glob(os.path.join(fake_dir, '**/*.mp4'), recursive=True)[:limit_per_class]
    for vid in fake_videos:
        visual_data.append(extract_face_frames(vid))
        audio_data.append(extract_mel_spectrogram(vid))
        labels.append(1) # 1 = Fake
        
    return np.array(visual_data), np.array(audio_data), np.array(labels)
