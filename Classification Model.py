#Classification Model

# Define the path to your audio dataset
dataset_path = '/kaggle/input/hackathon-dataset/Hackathon Dataset'
classes = ['normal', 'prolongation', 'repetition', 'blocking']

# Initialize empty lists to store features and labels
X = []
y = []

# Extract features from audio files
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for audio_file in os.listdir(class_path):
        audio_path = os.path.join(class_path, audio_file)
        
        y1, sr1 = librosa.load(audio_path)
        # Load audio file and extract features (you may need to install librosa)
        # chroma = librosa.feature.chroma_stft(S=spectrogram1, sr=sr1)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)
        spectral_centroid = librosa.feature.spectral_centroid(y=y1, sr=sr1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y1)
        duration = librosa.get_duration(y=y1, sr=sr1)
        duration = [duration]
        amplitude = np.max(np.abs(y1))
        amplitude = [amplitude]

        # Combine extracted features into a single feature vector
        combined_features = np.concatenate([mfccs.mean(axis=1), spectral_bandwidth.mean(axis=1), spectral_centroid.mean(axis=1),
                                           zero_crossing_rate.mean(axis=1), duration, amplitude])

        # Append feature vector and label to X and y lists
        X.append(combined_features)
        y.append(class_name)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Perform class balancing using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, shuffle=True, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Random Forest classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf_classifier = RandomForestClassifier(random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search
best_rf_classifier = grid_search.best_estimator_

# Train the best classifier on the training data
best_rf_classifier.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = best_rf_classifier.predict(X_test)
