# Dataset Card: Sign Language Motion Keypoint Corpus (SL-98)

## 1. Dataset Overview
- **Dataset Name**: Sign Language Motion Keypoint Corpus (SL-98)
- **Domain**: Assistive Technology / Deaf and Hard of Hearing (DHH) Human-Computer Interaction
- **Curated By**: Deep Learning Laboratory (Aligned to CSAI3027P)
- **Dataset Modality**: Multimodal (RGB video clips, extracted raw image frames, and 3D MediaPipe hand skeletal landmark tensors)
- **Total Samples**: 4,740 sequences
- **Total Categories**: 98 sign vocabulary classes (including everyday assistive commands, greetings, verbs, and questions)
- **Temporal Resolution**: 25 frames per sequence
- **Feature Dimension**: 126 floating-point values per frame (2 hands &times; 21 joints &times; 3 Cartesian coordinates: $(x, y, z)$)

---

## 2. Dataset Structure and Formats
The primary training, validation, and test datasets are stored as native NumPy binary arrays:
- `X_train_ready.npy`: Shape `(4740, 25, 126)`, dtype `float64`
- `y_train_ready.npy`: Shape `(4740,)`, dtype string (`classes.npy` vocabulary)
- `classes.npy`: 98 unique sign gloss strings (e.g., `afternoon`, `angry`, `answer`, `apple`, `baby`, `bad`, `bank`, `drink`, `go`, `help`, `welcome`)
- `SL/`: Directory containing raw video clips (`.mp4`) partitioned by gloss.
- `frame/`: Directory containing extracted high-resolution JPEG frames for 2D/3D visual inspection.

---

## 3. Data Governance & Leakage-Safe Splitting
To adhere strictly to industry-grade data science practices and prevent temporal/subject leakage:
- **Partition Ratio**: 70% Train (3,318 samples), 15% Validation (711 samples), 15% Test (711 samples).
- **Stratification**: All 98 classes are stratified across splits to preserve class distribution.
- **Normalization**: Hand landmark coordinates are zero-centered around the wrist (Joint 0) and scaled by palm span, ensuring invariance to camera distance and signer body frame differences.
- **Reproducibility**: Global seed `42` is fixed in `SignDatasetManager`.

---

## 4. Ethical, Sociotechnical, and Accessibility Considerations
1. **Assistive Utility**: Built to bridge communication barriers between the Deaf/Hard of Hearing community and hearing non-signers.
2. **Privacy Preservation**: Landmark representations abstract away facial and biometric identifiers, ensuring privacy compliance for camera-based continuous assistive monitors.
3. **Regional Dialect Sensitivity**: Acknowledges that sign languages (e.g., ASL, ISL, BSL) possess diverse dialects and cultural nuances; models trained on this vocabulary are assistive tools, not replacements for certified human interpreters.
4. **License**: Academic and research educational use permitted under the CSAI3027P Deep Learning curriculum framework.
