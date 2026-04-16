**Brain Tumor Detection and Segmentation from MRI Scans Using Deep Learning**
A two-stage deep learning pipeline that automatically detects and localizes brain tumors from MRI scans using ResNet50 for classification and ResUNet for segmentation.

**Overview**
Manual analysis of MRI scans is time-consuming and prone to human error. This project automates the process using a two-stage deep learning approach:\

Stage 1 — Classification: A fine-tuned ResNet50 model detects whether a tumor is present in the MRI scan.\
Stage 2 — Segmentation: A custom ResUNet model precisely localizes and delineates the tumor region at the pixel level.


**Dataset**
Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation \
Source: LGG MRI Segmentation Dataset — Kaggle\
Brain MRI images paired with ground truth tumor masks\
Binary labels: 0 = No Tumor, 1 = Tumor


**Model Architecture**
Stage 1 — ResNet50 Classifier

Pretrained on ImageNet\
Custom classification head with Dense, Dropout layers\
Output: 2-class softmax (tumor / no tumor)\
Loss: Categorical Cross-Entropy\
Optimizer: Adam

Stage 2 — ResUNet Segmentation

Hybrid of ResNet residual blocks + UNet encoder-decoder structure\
Skip connections for precise spatial localization\
Output: Per-pixel sigmoid (tumor mask)\
Loss: Focal Tversky Loss (α = 0.7)\
Optimizer: Adam (lr = 0.05, epsilon = 0.1)


**Evaluation Metrics**
Classification (ResNet50)\
MetricFormulaAccuracy(TP + TN) / (TP + TN + FP + FN)SensitivityTP / (TP + FN)SpecificityTN / (TN + FP)F1 Score2TP / (2TP + FP + FN)\
Segmentation (ResUNet) — Pixel Level\
MetricFormulaAccuracy(TP + TN) / (TP + TN + FP + FN)SensitivityTP / (TP + FN)SpecificityTN / (TN + FP)Dice / F12TP / (2TP + FP + FN)Tversky IndexTP / (TP + α·FN + (1−α)·FP)

**Project Structure**
├── medical_imaging.ipynb           # Main project notebook\
├── utilities.py                    # Custom data generator, loss functions, prediction\
├── ResUNet-segment-weights.keras   # Trained segmentation model weights\
├── classifier-resnet-weights.keras # Trained classifier model weights\
├── ResUNet-segment-model.json      # Segmentation model architecture\
├── classifier-resnet-model.json    # Classifier model architecture\
└── README.md

**Installation**
bashgit clone https://github.com/LlenGit/Medical_Image_processing\
cd brain-tumor-detection-segmentation\
pip install -r requirements.txt\
Requirements\
tensorflow\
numpy\
pandas\
opencv-python\
scikit-image\
scikit-learn\
matplotlib\
seaborn\
plotly\
Pillow\
keras_preprocessing

**Usage**

Download the dataset from Kaggle\
Place the dataset in the appropriate directory\
Run notebook.ipynb end to end\
Trained weights are saved automatically after training

To run inference on a single MRI image:\
python# Load models and run prediction\
image_id, mask, has_mask = prediction(test_df, model, model_seg)

**Results**
ModelMetricScoreResNet50:\
ClassifierAccuracy                : 0.9792%\
ResNet50 ClassifierSensitivity    : 0.9531%\
ResNet50 ClassifierSpecificity    : 0.9922%\
ResNet50 ClassifierF1 Score       : 0.9683%

Segment_ResNetScore:\
Pixel-level Accuracy    : 0.9928\
Sensitivity             : 0.9431\
Specificity             : 0.9934\
F1 Score (Dice)         : 0.7582\
Tversky Index (α=0.7)   : 0.8227

<img width="1014" height="470" alt="image" src="https://github.com/user-attachments/assets/91b1000f-4d1c-4659-bc0a-85da2b7a0bc9" />
<img width="1012" height="465" alt="image" src="https://github.com/user-attachments/assets/90c93212-faa3-478b-98af-a10278ab34a4" />



**Loss Function**
The segmentation model uses Focal Tversky Loss, which penalizes false negatives (missed tumor pixels) more than false positives — critical in medical imaging where missing a tumor is more dangerous than a false alarm.\
Tversky=TPTP+α⋅FN+(1−α)⋅FP,α=0.7\text{Tversky} = \frac{TP}{TP + \alpha \cdot FN + (1-\alpha) \cdot FP}, \quad \alpha = 0.7Tversky=TP+α⋅FN+(1−α)⋅FPTP​,α=0.7\
Focal Tversky Loss=(1−Tversky)γ,γ=0.75\text{Focal Tversky Loss} = (1 - \text{Tversky})^{\gamma}, \quad \gamma = 0.75Focal Tversky Loss=(1−Tversky)γ,γ=0.75\

**Acknowledgements**

Dataset: Mateusz Buda — Kaggle\
Focal Tversky Loss: nabsabraham/focal-tversky-unet\
ResNet Paper: Deep Residual Learning for Image Recognition

*Academic Use Only*
All required external links are given in the main notebook.\
Built as part of a thesis project on deep learning applications in medical imaging.
