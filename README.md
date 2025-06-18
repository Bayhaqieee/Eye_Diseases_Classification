
# Eye Diseases Classification using DenseNet-161

## 📌 What are you making?
This project aims to build an image classification model to detect and classify **four types of eye diseases** from retinal images:
- **Cataract**
- **Diabetic Retinopathy**
- **Glaucoma**
- **Normal**

The purpose of this project is to assist in early detection and diagnosis of eye diseases using deep learning and computer vision.

---

## 🏗️ What architecture you use?
The model is based on **DenseNet-161**, a convolutional neural network (CNN) architecture pre-trained on the ImageNet dataset. The final classification layer has been replaced with a custom `nn.Linear` layer to fit the 4-class classification task.

---

## 🧪 What library you use?
This project is implemented using the following libraries:
- `PyTorch`
- `Torchvision`
- `NumPy`
- `Matplotlib`
- `tqdm`
- `LazyPredict` (for model comparison)
- `scikit-learn` (used indirectly in LazyPredict)
- `TensorBoard`
- `os`, `shutil`, `random`, and other Python standard libraries

---

## 🛠️ How to run your model

### 1. Prepare the Dataset
Download and unzip the dataset into the following structure:
```
/dataset/
├── Cataract/
├── Diabetic Retinopathy/
├── Glaucoma/
└── Normal/
```

### 2. Split Dataset (Train/Validation)
```python
# Split 90% training and 10% validation using shutil and os modules
```

### 3. Data Preprocessing
```python
# Resize images to 224x224
# Normalize using ImageNet statistics
# Use torchvision.transforms.Compose for train and val sets
```

### 4. Model Setup
```python
# Load DenseNet161 from torchvision.models
# Replace classifier with nn.Linear(num_ftrs, 4)
```

### 5. Training
```python
# Use SGD optimizer with lr=0.0005 and momentum=0.9
# Scheduler: StepLR with step_size=6, gamma=0.1
# Loss function: CrossEntropyLoss
# Epochs: 20
```

### 6. Evaluation
```python
# Evaluate on validation data using accuracy metrics
# Visualize predictions and confusion matrix
```

---

## 📊 Model Baseline Comparison (Using LazyPredict)

Before switching to CNNs, we evaluated traditional ML models using feature vectors and LazyPredict. Below are some top-performing models:

| Model                         | Accuracy | F1 Score | Time Taken |
|------------------------------|----------|----------|-------------|
| **LGBMClassifier**            | 0.70     | 0.69     | 3.28 s      |
| **PassiveAggressiveClassifier** | 0.68  | 0.67     | 0.32 s      |
| **XGBClassifier**             | 0.68     | 0.66     | 4.63 s      |
| **Perceptron**                | 0.68     | 0.67     | 0.16 s      |
| **DecisionTreeClassifier**    | 0.62     | 0.63     | 0.25 s      |

While some traditional classifiers performed decently, the DenseNet161 CNN model achieved **significantly better generalization and performance**.

---

## 📊 Dataset Reference
The dataset is taken from Kaggle: [Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

It contains thousands of retinal images categorized into:
- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

---

## 📚 Paper Research Reference
- Huang, Gao, et al. *Densely connected convolutional networks*. CVPR 2017.
- Gulshan, Varun, et al. *Development and validation of a deep learning algorithm for detection of diabetic retinopathy*. JAMA, 2016.

---

## ✅ Result
- **Training completed in:** 35m 41s
- **Best Validation Accuracy:** 93.38%
- **Model architecture:** DenseNet-161 fine-tuned on custom dataset

---


