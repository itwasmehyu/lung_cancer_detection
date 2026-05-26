# Lung Cancer Detection using Convolutional Neural Network

A deep learning project for automated lung cancer classification using medical imaging. This project implements a custom CNN to classify lung tissues into three categories: normal, adenocarcinoma, and squamous cell carcinoma.

## Overview

This project uses a Convolutional Neural Network (CNN) to detect and classify lung cancer from medical images. The model is trained to distinguish between three types of lung tissues:

- **Normal (lung_n)**: Healthy lung tissue
- **Adenocarcinoma (lung_aca)**: Most common type of lung cancer
- **Squamous Cell Carcinoma (lung_scc)**: Common type of lung cancer

## Dataset

The dataset is organized into three subdirectories under `lung_subset_small/`:

```
lung_subset_small/
├── lung_n/       # Normal lung images
├── lung_aca/     # Adenocarcinoma images
└── lung_scc/     # Squamous cell carcinoma images
```

**Note**: The `lung_subset_small/` folder is excluded from version control (see `.gitignore`).

## Model Architecture

The project implements a custom Sequential CNN with the following structure:

- **Input**: 128×128 RGB images
- **Convolutional Block 1**: Conv2D(32 filters, 5×5) + ReLU + MaxPooling(2×2)
- **Convolutional Block 2**: Conv2D(64 filters, 3×3) + ReLU + MaxPooling(2×2)
- **Convolutional Block 3**: Conv2D(128 filters, 3×3) + ReLU + MaxPooling(2×2)
- **Fully Connected Layers**:
  - Dense(256) + ReLU + BatchNormalization
  - Dense(128) + ReLU + Dropout(0.3) + BatchNormalization
  - Dense(3) + Softmax (output layer)

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Image Size**: 128×128 pixels
- **Epochs**: 10 (with early stopping)
- **Validation Split**: 20%

### Callbacks

- **EarlyStopping**: Monitors validation accuracy with patience of 3 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 0.5x when validation loss plateaus
- **Custom Callback**: Stops training when validation accuracy reaches 90%

## Requirements

- Python 3.7+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- OpenCV (cv2)
- Scikit-learn

Install dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib pillow opencv-python scikit-learn
```

## Usage

1. **Prepare Data**: Extract the dataset to the `lung_subset_small/` directory

2. **Run the Notebook**: Open and execute `Lung_Cancer_Detection_using_Convolutional_Neural_Network.ipynb`

   ```bash
   jupyter notebook Lung_Cancer_Detection_using_Convolutional_Neural_Network.ipynb
   ```

3. **Model Training**: The notebook will:
   - Load and preprocess images
   - Train the CNN model
   - Display training history plots
   - Generate a classification report

## Output

After training, the model provides:

- **Accuracy and loss plots**: Shows training vs validation performance
- **Classification Report**: Precision, recall, and F1-score for each class
- **Model Summary**: Layer information and parameter count

## Project Structure

```
lung_cancer_detection/
├── Lung_Cancer_Detection_using_Convolutional_Neural_Network.ipynb
├── lung_subset_small/          (ignored in git)
│   ├── lung_n/
│   ├── lung_aca/
│   └── lung_scc/
├── .gitignore
└── README.md
```

## Results

The model is trained with early stopping when validation accuracy exceeds 90%, ensuring efficient training and preventing overfitting.

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, Pillow
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Evaluation**: Scikit-learn

## Notes

- Images are rescaled to 128×128 pixels for model input
- Categorical cross-entropy is used for multi-class classification
- Data is split 80% training, 20% validation
- The custom callback monitors validation accuracy and stops training early if performance targets are met

## Future Improvements

- Implement data augmentation techniques
- Test with pre-trained models (ResNet, VGG, etc.)
- Add test set evaluation
- Implement cross-validation
- Explore ensemble methods
- Deploy model as a web application

---

**Author**: Lung Cancer Detection Project  
**Last Updated**: May 2026
