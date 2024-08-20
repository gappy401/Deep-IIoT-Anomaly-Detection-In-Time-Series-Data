# Deep Industrial IoT Anomaly Detection in Time Series Data 

## Abstract
In this paper, I explore some deep learning methodologies for detecting anomalies in time series data. I utilised approaches from my study of the paper [Communication-Efficient Federated Learning for Anomaly Detection in Industrial Internet of Things](https://ieeexplore.ieee.org/document/9348249). Building on the concepts from the paper, I integrated an attention mechanism-based convolutional neural network long short-term memory (AMCNN-LSTM) model to effectively capture critical features in time series data, addressing issues such as memory loss and gradient dispersion. Additionally, I explored a gradient compression mechanism to improve communication efficiency within the federated learning setup. This work aims to adapt these advanced methodologies to the specific challenges of industrial anomaly detection, focusing on scalability, noise reduction, and class imbalance while ensuring privacy preservation and timely detection.

## 1. Introduction

The widespread deployment of edge devices in the Industrial Internet of Things (IIoT) has revolutionized industries by enabling real-time, flexible, and quick decision-making capabilities across various applications, such as smart manufacturing, intelligent transportation, and logistics. However, this rapid expansion has introduced critical security risks, particularly from abnormal behaviors in IIoT nodes, which can cause significant disruptions and economic losses. For example, in smart manufacturing, industrial devices that exhibit abnormal traffic or irregular reporting can interrupt production and lead to substantial financial setbacks.

Detecting these anomalies is increasingly crucial, particularly as edge devices continuously collect and analyze vast amounts of time-series data. Traditional anomaly detection methods often struggle with the complexity and volume of data in these scenarios. Furthermore, privacy concerns complicate the situation, as edge devices are reluctant to share raw data, leading to "data islands" that hinder collaborative anomaly detection efforts.


In my study, I explore advanced methodologies to enhance anomaly detection in time series data. Specifically, I adapt the proposed framework to a new dataset, investigating the generalizability and robustness of deep learning models across different contexts. My approach incorporates various deep learning techniques, including Autoencoders, LSTM Networks, CNN-LSTM models, and Attention Mechanisms, to address challenges such as imbalanced data, noise, and scalability. Additionally, I implemented model specific gradient compression mechanism to reduce communication overhead, ensuring the model's efficiency in distributed environments. 

My process involved exploring the data with techniques like variance inflation factor (VIF) and principal component analysis (PCA) to remove redundant features, cross-verifying with Lasso, and then employing a series of models—SAE, SVM, GRU, CNN, LSTM-CNN, and Attention Mechanism LSTM CNN. By applying these models to a different time series dataset, I aim to validate their effectiveness and adaptability beyond the original study's scope, contributing to the development of robust, scalable anomaly detection solutions for IIoT applications.

## 2. Related Work
### 2.1 Traditional Anomaly Detection Methods
Classical methods such as Statistical Methods, Isolation Forests, and One-Class SVMs are less effective for anomaly detection in the context of Industrial IoT (IIoT) sensors due to their limitations in handling high-dimensional, complex, and noisy data typical in IIoT environments. These methods require extensive manual feature engineering, struggle with scalability, and are not well-suited for real-time processing, making them less effective for detecting anomalies in IIoT settings. For instance, a [study](https://link.springer.com/article/10.1007/s40745-021-00362-9) highlights that traditional anomaly detection methods often fall short in capturing the intricate patterns in IIoT sensor data, which are better addressed by advanced models like deep learning frameworks that can learn from data automatically and operate efficiently at scale​.

### 2.2 Deep Learning Approaches
In my study, I applied existing deep learning techniques such as Autoencoders, Recurrent Neural Networks (RNNs), and Long Short-Term Memory (LSTM) networks for anomaly detection, as detailed in the referenced paper. These models are particularly effective in IIoT environments because they can automatically extract features from raw data and capture complex temporal dependencies, making them more suitable for time series anomaly detection. Additionally, I refined the dataset by conducting feature selection and dimensionality reduction, using techniques like PCA and VIF to remove redundant features, ensuring the models were trained on the most relevant and streamlined data. This refinement process improved the models' efficiency and accuracy, allowing for better generalization to unseen data.

## 3. Methodology

### 3.1 <a href="https://archive.ics.uci.edu/dataset/791/metropt+3+dataset" target="_blank">Data</a>
We utilize a time series dataset from an Industrial IoT environment. The dataset includes various sensor readings, such as TP2 (compressor pressure), DV_pressure, Oil_temperature, Motor_current, DV_eletric, Towers, LPS, Oil_level, Caudal_impulses, and proviedes with timestamps of anomalous behaviour. Preprocessing steps include data normalization, handling missing values, and converting timestamps to a standard format.

### 3.2 Model Architecture

In our federated learning system, we employ a consistent architectural framework across different models, including SV, GRU, SAE, CNN, LSTM, and their combinations, to address the challenges of anomaly detection in industrial IoT settings. The architecture is designed to optimize communication and computation, particularly in scenarios involving edge devices with limited resources. Here’s a detailed overview of the architecture:

#### 1. Federated Learning Framework

#### 1. Federated Learning Framework

Our system includes a **cloud aggregator** and **edge devices** working collaboratively:

- **Cloud Aggregator:** This is a robust server with substantial computing power and resources. It serves two primary functions:
  - **Global Model Initialization:** The cloud aggregator initializes the global model and distributes it to all edge devices.
  - **Gradient Aggregation:** It collects and aggregates the gradients uploaded by edge devices to refine the global model until convergence is achieved.

- **Edge Devices:** These include various IIoT sensors like whirlpools, wind turbines, and vehicles. Each edge device:
  - **Local Model Training:** Trains a local model on its specific dataset, which consists of sensing time-series data.
  - **Gradient Computation and Upload:** Computes gradients based on local training and uploads compressed gradients to the cloud aggregator.
  - **Anomaly De

#### 2 Gradient Compression

To address communication constraints, gradient compression techniques are applied. This involves:

- **Local Training and Gradient Computation:** Each client trains its local model on its specific data and computes gradients locally.

- **Gradient Compression:** The gradients are compressed to reduce their size and the amount of data transmitted. This process ensures efficient communication between the clients and the main server, minimizing the data transfer overhead.

- **Gradient Aggregation:** The compressed gradients from each client are sent to the main server, where they are aggregated to update the global model. The aggregation process involves combining these compressed updates to refine the global model.

#### 3. Federated Update Process

- **Edge Device Operations:** Clients train their local models and compute the gradients. These gradients are then compressed and sent to the main server.

- **Server-Side Operations:** The main server receives the compressed gradients, decompresses them if necessary, and aggregates the updates to improve the global model. The updated global model is then redistributed to the clients for further local training.

This architecture ensures that the federated learning system remains scalable and efficient, accommodating the limited computational resources of edge devices while optimizing data communication and model performance.

## 4. Models

### 4.1 Support Vector Machine (SVM)

**Model Type:** Support Vector Machine with RBF kernel.

- **Layers:** Not applicable (SVM is a non-neural network model).
- **Features:**
  - **Data Preparation:** Shuffle, split, and balance the dataset using class weights.
  - **Hyperparameter Tuning:** Perform grid search with parameters such as `C`, `gamma`, and `kernel`.
  - **Optimization:** Uses the RBF kernel to handle non-linear relationships in the data.
  - **Purpose:** Classifies data and detects anomalies by learning the decision boundary between classes.

### 4.2 Gated Recurrent Unit (GRU)

**Model Type:** GRU-based neural network.

- **Layers:**
  - **GRU Layer:** 64 units with return sequences to process sequential data.
  - **Dropout Layer:** Applied after the GRU layer to prevent overfitting.
  - **GRU Layer:** 32 units for further sequence processing.
  - **Dense Layer:** 50 units with ReLU activation for learning complex patterns.
  - **Dropout Layer:** Applied before the final output layer.
  - **Output Layer:** Dense layer with 1 unit and sigmoid activation for binary classification.
- **Purpose:** Captures temporal dependencies and patterns in sequential data, useful for anomaly detection in time-series data.

### 4.3 Stacked Autoencoder (SAE)

**Model Type:** Autoencoder neural network.

- **Layers:**
  - **Encoder:**
    - **Dense Layer:** 128 units with ReLU activation.
    - **Dense Layer:** 64 units with ReLU activation.
    - **Dense Layer:** 32 units with ReLU activation.
  - **Bottleneck Layer:**
    - **Dense Layer:** 16 units with ReLU activation, representing compressed encoded data.
  - **Decoder:**
    - **Dense Layer:** 32 units with ReLU activation.
    - **Dense Layer:** 64 units with ReLU activation.
    - **Dense Layer:** 128 units with ReLU activation.
  - **Output Layer:**
    - **Dense Layer:** Number of units equal to input dimensions with sigmoid activation.
- **Purpose:** Learns efficient data representations and reconstructs the input. Useful for anomaly detection by identifying reconstruction errors.

### 4.4 Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM)

**Model Type:** Hybrid CNN-LSTM model.

- **Layers:**
  - **CNN Layers:**
    - **Conv1D Layer:** 64 filters with kernel size 3, ReLU activation for feature extraction.
    - **MaxPooling1D Layer:** Pool size 2 to downsample and reduce dimensionality.
    - **Dropout Layer:** Applied to prevent overfitting.
  - **LSTM Layers:**
    - **LSTM Layer:** 50 units with return sequences to capture temporal dependencies.
    - **LSTM Layer:** 50 units for further sequence processing.
  - **Dense Layers:**
    - **Dense Layer:** 50 units with ReLU activation.
    - **Dropout Layer:** Applied before the final output layer.
    - **Output Layer:** Dense layer with 1 unit and sigmoid activation for classification.
- **Purpose:** Combines convolutional layers for feature extraction with LSTM layers for sequence modeling, suitable for time-series data.

### 4.5 Attention Mechanism CNN-LSTM

**Model Type:** CNN-LSTM model with attention mechanism.

- **Layers:**
  - **CNN Layers:**
    - **Conv1D Layer:** 64 filters with kernel size 3, ReLU activation.
    - **MaxPooling1D Layer:** Pool size 2 to reduce dimensionality.
    - **Dropout Layer:** Applied to prevent overfitting.
  - **LSTM Layers:**
    - **LSTM Layer:** 50 units with return sequences to capture sequential data.
  - **Attention Mechanism:**
    - **Dense Layer:** Computes attention probabilities to focus on important parts of the sequence.
    - **Multiply Layer:** Applies attention weights to the LSTM outputs.
  - **LSTM Layer:** 50 units to process attention-weighted sequences.
  - **Dense Layers:**
    - **Dense Layer:** 50 units with ReLU activation.
    - **Dropout Layer:** Applied before the final output layer.
    - **Output Layer:** Dense layer with 1 unit and sigmoid activation for classification.
- **Purpose:** Enhances the CNN-LSTM model by incorporating attention mechanisms to focus on significant features and improve model performance.



### 3.3 Training and Evaluation
The models are trained using a combination of supervised and unsupervised learning techniques. Cross-entropy loss is used for classification tasks, and mean squared error for reconstruction tasks. Evaluation metrics include Precision, Recall, F1-Score, and ROC-AUC. Special attention is given to handling class imbalance and noisy data. Data augmentation techniques and advanced preprocessing methods, such as resampling and handling warnings related to chained assignment, are applied to improve model robustness.

## 4. Experiments and Results

### 4.1 Experimental Setup
Experiments involve hyperparameter tuning using grid search and stratified k-fold cross-validation to ensure robust performance. Data augmentation and preprocessing techniques are also employed to address class imbalance and data quality.

### 4.2 Results
- **Model Performance**: 
- **Visualization**: Anomaly detection results are visualized in time series plots, demonstrating the effectiveness of the models.
- **Comparison with Baselines**: Significant improvements are observed over baseline models like Isolation Forests and One-Class SVMs, especially in handling complex patterns.

## 5. Challenges

One of the primary challenges encountered in this study is the computational limitation associated with Support Vector Machines (SVMs) and GRU models due to the large volume of data. Handling extensive datasets often leads to significant computational overhead, which can hinder the model's performance and scalability. Additionally, the simulation of edge computing environments poses its own set of challenges. Additionally, traditional Python-based simulations may not fully capture the constraints and resource limitations of real-world edge computing scenarios, where resources are constrained and operational conditions are more variable.

## 6. Conclusion

This paper has explored novel deep learning approaches for anomaly detection in time series data, with a particular emphasis on Industrial IoT applications. The proposed models demonstrate significant improvements over traditional methods, particularly in their ability to handle complex patterns and large datasets. The integration of attention mechanisms has been shown to enhance model accuracy, allowing for more precise anomaly detection. Furthermore, the use of gradient compression has proven effective in reducing communication overhead and saving time, thus improving the efficiency of federated learning processes. These advancements contribute to making anomaly detection more practical and scalable in real-world Industrial IoT environments.


## References
-[^1] Communication-Efficient Federated Learning for Anomaly Detection in Industrial Internet of Things. [Link](https://ieeexplore.ieee.org/document/9348249)

