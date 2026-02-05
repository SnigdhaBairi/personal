Pneumonia Detection using Deep Learning

An AI-powered system for detecting pneumonia from chest X-ray images using Convolutional Neural Networks and transfer learning with VGG16.

Project Overview
This project implements a deep learning model to automatically classify chest X-rays as NORMAL or PNEUMONIA, achieving **93.59% accuracy** on test data.

Results
- Accuracy: 93.59%
- Precision: 94%
- Recall: 94%
- Dataset: 5,856 chest X-ray images

Technologies Used
- Python
- TensorFlow/Keras
- VGG16 (Transfer Learning)
- OpenCV
- Matplotlib & Seaborn

Dataset
Kaggle Chest X-Ray Pneumonia Dataset
- Training: 5,216 images
- Testing: 624 images
- Classes: NORMAL, PNEUMONIA

Key Features
- Transfer learning with VGG16 architecture
- Data augmentation for improved generalization
- Comprehensive performance visualizations
- Real-time prediction on new X-ray images

Model Architecture
- Base: VGG16 (pre-trained on ImageNet)
- Fine-tuned last 4 convolutional layers
- Custom classification head
- Binary classification output

 How to Run
1. Clone the repository
2. Install dependencies: `pip install tensorflow opencv-python matplotlib seaborn`
3. Open and run `pneumonia_detection.ipynb`

 Key Achievements
Built an AI system that detects pneumonia from chest X-rays with over 93% accuracy using transfer learning. The model provides confidence scores and visual metrics to help understand diagnosis reliability, making it a practical tool for assisting healthcare professionals.

Future Improvements
- Multi-class classification (bacterial vs viral)
- Web deployment for real-time predictions
- Integration with hospital PACS systems

Why I Built This
During my undergraduate studies, I became interested in how AI could assist in medical diagnosis. This project was my exploration into applying deep learning to real-world healthcare problems.
   
Challenges Faced
- Initial model overfitting (solved with dropout and data augmentation)
- Class imbalance in dataset (handled with careful evaluation metrics)
- Long training times (optimized with transfer learning)
