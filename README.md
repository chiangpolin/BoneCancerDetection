# BoneScanClassifier
Visit [Website](https://scan.chiangpolin.com/)

## Method
BoneScanClassifier is a deep learning-based image classification model designed to detect bone cancer from X-ray images.  
- The model is implemented using **PyTorch** and trained on a large labeled dataset of bone X-rays.  
- Images are preprocessed and augmented to improve generalization.  
- The model predicts whether an X-ray shows **cancer** or **normal** bone tissue.  
- For deployment, the trained model can be served via **FastAPI** for API-based inference.

## Dataset
- Source: https://www.kaggle.com/datasets/ziya07/bone-cancer-detection-dataset
- Contains **8,811 X-ray images** labeled for bone cancer detection.  
- The dataset is split into **train, test, and validation** subsets.  

### Directory structure
```
├── backend/
├── data/
│   ├── test/       
│   |   └── _classes.csv  
│   ├── train/        
│   |   └── _classes.csv  
│   └── valid/
│       └── _classes.csv  
├── frontend/
├── model/
│   ├── static/      
│   └── src/
|       └── main.py
```

### CSV format
Each CSV contains the following columns:
```
filename,cancer,normal
01.jpg,0,1
02.jpg,1,0
```
- `cancer` = 1 if the image shows cancer, 0 otherwise  
- `normal` = 1 if the image is normal, 0 otherwise  

## Model
- Model architecture: Convolutional Neural Network (CNN) based on [ResNet18 / Custom CNN]  
- Input: 224x224 X-ray images  
- Output: Binary classification (**cancer** or **normal**)  

**Training settings:**  
- Loss: Cross-Entropy Loss  
- Optimizer: Adam  
- Batch size: 16  
- Epochs: 10  

**Training Result:** 

![plot](/model/static/training_history.png)

**Confusion Matrix:** 

![plot](/model/static/confusion_matrix.png)

**ROC Curve:** 

![plot](/model/static/roc_curve.png)

