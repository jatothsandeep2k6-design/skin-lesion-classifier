# HYBRID MULTI-MODAL SKIN LESION CLASSIFICATION USING EFFICIENTNETB0 AND CLINICAL METADATA FUSION

---

## TITLE PAGE

**A PROJECT REPORT ON**

**HYBRID MULTI-MODAL SKIN LESION CLASSIFICATION USING
EFFICIENTNETB0 AND CLINICAL METADATA FUSION**

Submitted in partial fulfilment of the requirements
for the award of the degree of

**BACHELOR OF TECHNOLOGY**

in

**Computer Science and Engineering
(Artificial Intelligence & Machine Learning)**

Submitted by

**Jatoth Sandeep — 23VD1A6642**

Under the guidance of

**[Guide Name], [Designation]**

Department of Computer Science and Engineering (AI & ML)

**JNTUH University College of Engineering Manthani**

Academic Year: 2023–2027

---

## ABSTRACT

Skin cancer is among the most common malignancies worldwide,
and early, accurate diagnosis is critical to improving patient
survival rates. Visual inspection by dermatologists, while
effective, is subject to inter-observer variability and is not
uniformly accessible, particularly in low-resource and rural
healthcare settings. This project presents a hybrid multi-modal
deep learning system for the automated classification of
dermoscopic skin lesion images into seven diagnostic categories,
combining convolutional image features with structured clinical
metadata (patient age, sex, and lesion anatomical location) to
improve diagnostic reliability over image-only approaches.

The proposed architecture employs EfficientNetB0, a
compound-scaled convolutional neural network pretrained on
ImageNet, as the image encoding branch, fine-tuned using
transfer learning on the HAM10000 (Human Against Machine with
10000 training images) dataset. A parallel dense-layer branch
processes the three clinical metadata features. The two branches
are fused through feature concatenation followed by fully
connected classification layers, allowing the network to jointly
reason over visual texture and patient context rather than image
content alone. Class imbalance, a significant challenge in
HAM10000 where the majority class (melanocytic nevi) outnumbers
the rarest class (dermatofibroma) by a factor of more than fifty,
is addressed using computed class weighting and targeted data
augmentation.

Beyond model development, this project delivers an interactive
web-based screening application built using Streamlit, through
which a clinician can upload a lesion image, enter patient
metadata, and receive a predicted diagnostic class, a confidence
score, a colour-coded clinical risk level, and an interactive
probability distribution across all seven classes. The system is
positioned explicitly as a clinical decision-support and
screening aid, not a replacement for professional dermatological
diagnosis.

**Index Terms:** Skin lesion classification, EfficientNetB0,
transfer learning, multi-modal fusion, HAM10000, convolutional
neural network, clinical decision support, Streamlit.

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Background

Skin cancer represents one of the most rapidly increasing cancer
diagnoses globally, encompassing both melanoma and non-melanoma
forms. Melanoma, although less common than other skin cancers,
accounts for a disproportionately high share of skin-cancer-related
deaths because of its potential to metastasize if not identified
early. Conversely, several non-malignant or pre-malignant lesion
types, such as melanocytic nevi and benign keratoses, visually
resemble cancerous lesions to the untrained eye, making accurate
classification a genuinely difficult diagnostic task even for
trained clinicians.

Dermoscopy, a non-invasive imaging technique that magnifies
subsurface skin structures, is the standard tool used by
dermatologists for lesion assessment. However, the diagnostic
accuracy of dermoscopic evaluation is heavily dependent on the
examiner's experience, and access to specialist dermatologists
is unevenly distributed, particularly in rural and under-resourced
regions. These constraints motivate the development of automated,
AI-assisted screening tools that can support — though not replace
— clinical decision-making by providing a fast, consistent, and
reproducible first-pass assessment of lesion images.

### 1.2 Motivation

Convolutional Neural Networks (CNNs) have demonstrated
dermatologist-level performance on several controlled skin lesion
classification benchmarks, largely driven by the availability of
large, well-annotated public datasets such as HAM10000 and the
ISIC archive. However, the majority of published models operate
on image data in isolation, discarding clinically relevant
non-image information that dermatologists routinely use during
diagnosis — most notably, a patient's age, sex, and the
anatomical location of the lesion.

This project is motivated by the hypothesis that fusing
image-derived features with structured clinical metadata can
produce a more clinically grounded classifier than an image-only
model, while remaining lightweight enough to train and deploy
without specialized infrastructure. EfficientNetB0 was selected
specifically because its compound scaling approach yields strong
accuracy-per-parameter efficiency, making it practical to
fine-tune on a free-tier GPU environment.

### 1.3 Problem Statement

Manual visual classification of dermoscopic skin lesion images
into diagnostic categories is subjective, inconsistent across
examiners, and not scalable to the volume of screening required
in large or under-served populations. There is a need for a
system that:

- Classifies dermoscopic images into clinically meaningful
  diagnostic categories with measurable, reproducible accuracy
- Incorporates available patient metadata (age, sex, lesion
  location) as a complementary signal alongside image features
- Addresses the severe class imbalance inherent in real-world
  dermatological datasets
- Is presented through an accessible interface that communicates
  a predicted label, confidence level, and clinical risk tier

### 1.4 Objectives

- To design and implement a multi-modal neural network
  architecture that fuses EfficientNetB0-derived image features
  with a dedicated clinical metadata branch for seven-class
  skin lesion classification
- To fine-tune the image branch using transfer learning on the
  HAM10000 dataset, employing a two-phase training strategy
- To mitigate class imbalance using computed class weighting
  and targeted data augmentation
- To evaluate the trained model using accuracy, precision,
  recall, F1-score, and confusion matrix analysis
- To develop an interactive Streamlit-based web application
  for clinical screening support
- To document the complete system clearly enough to be
  reproduced and extended by future students

### 1.5 Scope of the Project

The scope covers the end-to-end pipeline from raw dataset
acquisition to a functioning, locally deployable screening
application. This includes dataset preprocessing, model
architecture design, two-phase model training, quantitative
evaluation, and the development of a user-facing Streamlit
interface.

The project does not extend to clinical trial validation,
regulatory certification as a medical device, or cloud
deployment for multi-user concurrent access.

### 1.6 Limitations

- The model is trained exclusively on HAM10000
- Class imbalance, while mitigated, is not fully eliminated
- Only three metadata fields are used (age, sex, location)
- The application is strictly a screening and educational aid

### 1.7 Organization of the Report

- Chapter 2: Literature Survey
- Chapter 3: Existing System Analysis
- Chapter 4: Proposed System
- Chapter 5: Methodology
- Chapter 6: System Architecture
- Chapter 7: Implementation Details
- Chapter 8: Results and Discussion
- Chapter 9: Web Application
- Chapter 10: Future Scope
- Chapter 11: Conclusion
- References

---

## CHAPTER 2 — LITERATURE SURVEY
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 3 — EXISTING SYSTEM
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 4 — PROPOSED SYSTEM
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 5 — METHODOLOGY
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 6 — SYSTEM ARCHITECTURE
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 7 — IMPLEMENTATION
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 8 — RESULTS AND DISCUSSION
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 9 — WEB APPLICATION
[PASTE HERE when remaining docs arrive]

---

## CHAPTER 10 — FUTURE SCOPE

### 10.1 Gradient-Weighted Class Activation Mapping (Grad-CAM)
The current system produces a predicted class and confidence
score, but does not communicate which regions of the uploaded
image most influenced the prediction. Integrating Grad-CAM
would allow a clinician to verify whether the model is
attending to the lesion itself rather than peripheral artefacts.

### 10.2 Vision Transformer (ViT) Architecture
A future version could replace EfficientNetB0 with a
lightweight ViT variant such as DeiT-Small or MobileViT,
enabling the model to capture long-range spatial dependencies
across the entire image through self-attention mechanisms.

### 10.3 Extended Metadata and Patient History Integration
Future work could increase the metadata branch input from
three features to include Fitzpatrick skin type, lesion
evolution history, UV exposure, and family history of cancer.

### 10.4 Mobile Application Deployment
Converting the model to TensorFlow Lite format would enable
on-device inference on Android and iOS hardware without
requiring a network connection.

### 10.5 Cloud Deployment and Multi-User Scalability
Deploying on Google Cloud Run, AWS, or Azure would allow
multiple clinicians to access the tool simultaneously through
a browser without any local installation.

---

## CHAPTER 11 — CONCLUSION

This project designed, implemented, and evaluated a hybrid
multi-modal deep learning system for the automated
classification of dermoscopic skin lesion images into seven
diagnostic categories.

The proposed architecture fuses EfficientNetB0-derived image
features with structured clinical metadata through feature
concatenation. Training followed a two-phase transfer learning
strategy with computed class weighting and data augmentation
to handle the 58:1 class imbalance in HAM10000.

**Key Results:**
- Overall Accuracy: 89.88%
- Macro-average AUC-ROC: 0.97
- Weighted-average F1-Score: 0.9003
- Best class: Melanocytic Nevi (F1 = 0.9489)
- Challenging class: Dermatofibroma (F1 = 0.7059)

The Streamlit web application delivers predictions in a
clinician-readable format with risk level, confidence score,
and probability chart. The system is positioned as a
screening and educational aid, not a replacement for
professional diagnosis.

---

## REFERENCES

[1] P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000
dataset, a large collection of multi-source dermatoscopic
images of common pigmented skin lesions," Scientific Data,
vol. 5, no. 1, p. 180161, Aug. 2018.

[2] M. Tan and Q. V. Le, "EfficientNet: Rethinking model
scaling for convolutional neural networks," in Proc. 36th
Int. Conf. Machine Learning (ICML), Long Beach, CA, USA,
Jun. 2019, pp. 6105–6114.

[3] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and
Z. Wojna, "Rethinking the inception architecture for
computer vision," in Proc. IEEE Conf. CVPR, 2016,
pp. 2818–2826.

[4] J. Kawahara, S. Daneshvar, G. Argenziano, and
G. Hamarneh, "7-Point checklist and skin lesion
classification using multitask multimodal neural nets,"
IEEE J. Biomed. Health Inform., vol. 23, no. 2,
pp. 538–546, Mar. 2019.

[5] N. C. F. Codella et al., "Skin lesion analysis toward
melanoma detection: ISIC 2018 challenge," arXiv preprint
arXiv:1902.03368, 2019.

[6] A. G. Howard et al., "MobileNets: Efficient convolutional
neural networks for mobile vision applications," arXiv
preprint arXiv:1704.04861, 2017.

[7] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning,"
Nature, vol. 521, no. 7553, pp. 436–444, May 2015.

[8] A. Krizhevsky, I. Sutskever, and G. E. Hinton,
"ImageNet classification with deep convolutional neural
networks," in NeurIPS, vol. 25, 2012, pp. 1097–1105.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual
learning for image recognition," in Proc. IEEE CVPR,
2016, pp. 770–778.

[10] A. Esteva et al., "Dermatologist-level classification
of skin cancer with deep neural networks," Nature,
vol. 542, no. 7639, pp. 115–118, Feb. 2017.

[11] D. P. Kingma and J. Ba, "Adam: A method for stochastic
optimization," in Proc. ICLR, San Diego, CA, 2015.

[12] N. Srivastava et al., "Dropout: A simple way to prevent
neural networks from overfitting," JMLR, vol. 15, no. 1,
pp. 1929–1958, 2014.

[13] S. Ioffe and C. Szegedy, "Batch normalization," in
Proc. ICML, 2015, pp. 448–456.

[14] R. R. Selvaraju et al., "Grad-CAM: Visual explanations
from deep networks," in Proc. IEEE ICCV, 2017,
pp. 618–626.

[15] A. Dosovitskiy et al., "An image is worth 16x16 words:
Transformers for image recognition at scale," in Proc.
ICLR, 2021.

[16] H.-C. Shin et al., "Deep CNNs for computer-aided
detection," IEEE Trans. Med. Imaging, vol. 35, no. 5,
pp. 1285–1298, May 2016.