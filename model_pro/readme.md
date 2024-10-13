# BCAT Model: Parallel Chinese Offensive Language Detection with Synergistic Semantics and Topic Modeling

## Overview

The BCAT (BERT-CTM Attention-based Text Classifier) model is designed for Chinese sentiment recognition, particularly focusing on offensive and aggressive language detection. The model leverages BERT-generated contextual word embeddings and CTM (Combined Topic Modeling) to capture both semantic and thematic features from text data. BCAT integrates a multi-head attention mechanism to enhance feature representation and applies convolutional networks (DPCNN and TextCNN) for feature extraction.

## Features

- **BERT and CTM Fusion**: BCAT effectively combines BERT embeddings with CTM topic vectors. BERT captures the context of words in a sentence, while CTM identifies overarching themes, improving the model's ability to detect nuanced sentiments.
- **Multi-Head Attention Mechanism**: This component focuses on different aspects of the input data, ensuring that critical features are emphasized during classification.
- **TextCNN and DPCNN**: These two convolutional networks operate in parallel to extract both local (TextCNN) and global (DPCNN) features, improving the robustness of the model for different linguistic structures.

## Model Architecture

The BCAT model is divided into the following components:

1. **Embedding Layer**: Text data is transformed into embeddings using the BERT model.

2. Feature Extraction Layer:

   - TextCNN extracts local features (word and phrase combinations).
   - DPCNN captures global text structure and long-range dependencies.
   
3. **Feature Fusion and Attention**: The output from both networks is combined and processed by the multi-head attention mechanism to highlight relevant information.

4. **Classification Layer**: A fully connected Softmax layer outputs the predicted sentiment classes.

## Data

BCAT is trained on the **COLD (Chinese Offensive Language Dataset)**, a publicly available dataset that includes offensive and safe comments across various categories. The model also uses real-time data collected from social platforms like Weibo through a custom web crawler.

### Dataset Statistics

- **COLD Dataset**: Contains 37,480 comments with binary labels indicating whether a comment is offensive or safe.
- **Weibo Data**: Supplementary real-world data gathered through a web crawler to ensure model robustness in practical applications.

## Training and Testing

- **Training**: The model was trained on a dataset split into training, validation, and test sets. Key metrics such as accuracy, precision, recall, and F1-score were used to evaluate performance.
- **Testing**: The model underwent extensive testing with the validation and test datasets, showing excellent results in offensive language detection.

### Key Performance Metrics

| Component Configuration                        | Precision | Recall | F1 Score |
|------------------------------------------------|-----------|--------|----------|
| BCAT (BERT + CTM + DPCNN + TextCNN + MHA)      | 89.35%    | 86.81% | 87.34%   |
| BERT + DPCNN + TextCNN + MHA                   | 87.85%    | 85.34% | 85.35%   |
| BERT + CTM + TextCNN + MHA                     | 86.66%    | 85.14% | 84.97%   |

## How to Use

1. **Dependencies**:

   - Python 3.8+
   - PyTorch
   - Transformers (Hugging Face)
   - Contextualized Topic Models (CTM)
   - Jieba for Chinese tokenization

2. **Installation**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Training**: To train the BCAT model with your own dataset:

   ```bash
   python train_model.py --data_path <path_to_data> --save_path <path_to_save_model>
   ```

4. **Inference**: For predicting sentiment on new text data:

   ```bash
   python predict.py --model_path <path_to_model> --input_text "Your input text here"
   ```