## Enhancing Monkeypox Detection through Ensemble Deep Transfer Learning Models

I am using the dataset from https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20

**Monkeypox Skin Lesion Dataset Version 2.0 (MSLD v2.0)** comprises images from six distinct classes, namely Mpox (284 images), Chickenpox (75 images), Measles (55 images), Cowpox (66 images), Hand-foot-mouth disease or HFMD (161 images), and Healthy (114 images). The dataset includes 755 original skin lesion images sourced from 541 distinct patients, ensuring a representative sample. Importantly, the latest version has received endorsement from professional dermatologists and obtained approval from appropriate regulatory authorities.

Content
The dataset is organized into two folders:

Original Images: This folder includes a subfolder named "FOLDS" containing five folds (fold1-fold5) for 5-fold cross-validation with the original images. Each fold has separate folders for the test, train, and validation sets.

Augmented Images: To enhance the classification task, various data augmentation techniques, such as rotation, translation, reflection, shear, hue, saturation, contrast, brightness jitter, noise, and scaling, were applied using MATLAB R2020a. To ensure result reproducibility, the augmented images are provided in this folder. It contains a subfolder called "FOLDS_AUG" with augmented images of the train sets from each fold in the "FOLDS" subfolder of the "Original Images". The augmentation process resulted in an approximate 14-fold increase in the number of images.

Naming Convention of the Images
Each image is assigned a name following the format of DiseaseCode_PatientNumber_ImageNumber. The corresponding disease codes assigned to each of the six disease classes are - Mpox -> MKP, Chickenpox -> CHP, Cowpox -> CWP, Measles -> MSL, Hand,foot and mouth disease -> HFMD, Healthy -> HEALTHY. Assignment of the keywords is illustrated in the provided image "Keywords.jpg".
For instance, an image named "MKP_17_01" indicates that it belongs to the Mpox class and is the first image captured from a patient with the ID 17.

Data organization
The dataset includes an Excel file named "datalog.xlsx" consisting of 5 sheets (Sheet1-5), with each sheet corresponding to a specific fold (fold1-5). Each sheet contains three columns: train, validation, and test. These columns contain the names of the images belonging to the respective train, validation, and test sets for a particular fold.

## Technologies Used:
Python, Machine Learning, Deep Learning, CNN, Deep Transfer Learning

## STAGES OF PROPOSED MODEL
In fig.1, we have a proposed Ensemble Deep Transfer Learning Models which mainly  consists of 5 Stages for efficient and accurate classification of the monkeypox. The 5 stages 
are as follows:
1. Image Preprocessing.
  • Image Data Augmentation
  • Data Partitioning
2. Training and Testing of Deep Transfer Learning -based Models
3. Implementing Ensemble Learning Classifiers like Hard Voting and Soft Voting.
4. Implementation of Combined Voting
5. Apply Post-processing technique
6. Output/ Diagnostic Result

**Image preprocessing**: Image pre-processing refers to a set of techniques used to  prepare digital images for further analysis or processing. This is commonly done in fields like 
computer vision, medical imaging, and satellite image processing. Here are some common steps involved in image pre-processing:
• Scaling Features: The skin photos in the dataset are first convertedto RGB using OpenCV techniques, and then they are resized to 224 × 224 pixels. The maximum picture size that the system will accept is 255,
although the normal image size falls between [0,255]. 
• Data Augmentation for MSLDV2 Dataset: Data augmentation is the technique of adding more data to a dataset by transforming the existing data in an unpredictable way. The **Keras** deep learning system has a class called
**ImageDataGenerator** that lets us usepicture data to fit the model.
• Data splitting: In particular, the dataset was split into two halves for this study: 80% was used for training and 20% was used to evaluate our suggested model. For model validation, 20% of the picture 
samples from the training dataset are used.

**Training and Testing of Deep Transfer Learning -based Models**

**Steps for DTL models creation:**
1. **Pretrained Model Initialization:** We start by loading a pretrained Deep Transfer models such as DenseNet121, DenseNet169, InceptionResNetV2, ResNet152V2, DenseNet201, 
which has been trained on a large-scale dataset such as ImageNet. This pretrained model serves as the foundation for our customization process.
2. **Top Layer Removal:** The original classification layers of the all DTL models, responsible for predicting ImageNet classes, are removed. This step allows us to reconfigure the model's 
output to accommodate a different number of classes relevant to our target dataset.
3. **Feature Extraction with Global Average Pooling:** After removing the top layers, we introduce a Global Average Pooling (GAP) layer. This layer aggregates spatial information 
across the feature maps produced by the preceding layers, resulting in a compact representation of the input image.
4. **Flattening and Dense Layers:** The output of the GAP layer is flattened into a 1-dimensional vector to prepare it for input into fully connected dense layers. We add two hidden dense 
layers with Rectified Linear Unit (ReLU) activation functions, which introduce nonlinearity to the model. Dropout regularization is applied to mitigate overfitting by randomly 
deactivating a fraction of neurons during training.
5. **Output Layer Customization:** Finally, we append a dense output layer with softmax activation. This layer transforms the model's raw predictions into probabilities for each class 
in the target dataset.

Model Training and testing:
The model is assembled and then trained using the categorical cross-entropy loss function and the Adam optimizer (Kingma & Ba, 2014). The training dataset was utilized to train our model, and the validation dataset was used tovalidate it. The
accuracy measure is employed to evaluate the overall performance of the model. Ultimately, the test data is used to put the model through its paces.


**Implementing Ensemble Learning Classifiers like Hard Voting and Soft Voting.**
In an ensemble classifier, multiple individual classifiers (models) are combined to make predictions collectively, with the aim of improving overall performance compared to any single constituent classifier. Typically, each individual classifier 
in the ensemble provides its own probability estimates for each class in the classification task. These probabilities represent the confidence or certainty of the classifier regarding each possible class assignment for a given input instance.
When it comes to ensemble classifiers, there are two primary methods of combining the predictions from individual classifiers: hard voting and soft voting.

**Implementation of Combined Voting**
To combine hard and soft voting, you can first compute the hard voting predictions and then compute the soft voting predictions only for the cases where there is no 
clear majority in the hard voting predictions.By combining both hard and soft voting, you can leverage the benefits of both approaches and potentially improve the overall accuracy of your ensemble 
classifier.

• I've added a function combined_voting to perform ensemble voting with confidence score and threshold adjustment.
• Inside train_evaluate_ensemble, after evaluating individual models and obtaining their predictions, I've calculated ensemble predictions using hard voting, soft voting, and the combined approach.
• The combined_voting function takes the predictions from individual models, calculates the confidence scores, and then combines the predictions based on the threshold.
• Finally, I've evaluated the combined predictions and printed the classification report and confusion matrix for the combined model

**Apply Post-processing technique (Smoothing Technique)- Median Filtering**
Post-processing techniques can be applied after obtaining the combined predictions to refine the final output. In the provided code, we can apply postprocessing techniques such as filtering or smoothing to the predictions to remove 
noise or outliers and to improve the quality of the predictions. Smoothing and filtering are both techniques used in signal and image processing to enhance or modify data. While they serve similar purposes, they are typically 
used in slightly different contexts and may employ different mathematical methods.

• I've defined a function apply_post_processing to apply median filtering to the combined predictions.
• The apply_post_processing function takes the combined predictions and applies the median filter with a specified filter size (in this case, 3x3).
• After obtaining the combined predictions, we apply the post-processing technique to filter the predictions.
• Finally, we evaluate the filtered predictions and print the classification report and confusion matrix for the combined model after post-processing.






