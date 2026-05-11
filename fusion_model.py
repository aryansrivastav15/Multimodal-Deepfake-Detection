# ==========================================
# 2. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
def build_multimodal_model():
    visual_input = Input(shape=(224, 224, 3), name="visual_input")
    base_visual_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=visual_input)
    base_visual_model.trainable = False 
    visual_features = Flatten()(base_visual_model.output)
    visual_features = Dense(256, activation='relu')(visual_features)

    audio_input = Input(shape=(224, 224, 3), name="audio_input")
    x = Conv2D(32, (3, 3), activation='relu')(audio_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    audio_features = Dense(256, activation='relu')(x)

    merged = Concatenate()([visual_features, audio_features])
    z = Dense(512, activation='relu')(merged)
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    output = Dense(1, activation='sigmoid', name="output")(z)

    model = Model(inputs=[visual_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. EXECUTION: TRAINING & GRAPHING
# ==========================================
# ---> PASTE YOUR FOLDER PATHS HERE <---
REAL_VIDEO_DIR = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/original" 
FAKE_VIDEO_DIR = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/Deepfakes"

print("\n--- PHASE 1: DATA EXTRACTION ---")
X_visual, X_audio, y_labels = prepare_baseline_data(REAL_VIDEO_DIR, FAKE_VIDEO_DIR, limit_per_class=100)

print("\n--- PHASE 2: MODEL TRAINING ---")
multimodal_model = build_multimodal_model()
history = multimodal_model.fit(
    x=[X_visual, X_audio], 
    y=y_labels, 
    epochs=5,           
    batch_size=8,       
    validation_split=0.2 
)

print("\n--- PHASE 3: SAVING ARTIFACTS ---")
multimodal_model.save("multimodal_baseline.h5")
print("Model weights saved to multimodal_baseline.h5")
