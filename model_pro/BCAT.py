import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from CNN import extract_CNN_features
from MHA import MultiHeadAttentionLayer
from classifier import FinalClassifier
from BERT_CTM import BERT_CTM_Model
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# BERT_CTM embeddings generation and loading function
def get_bert_ctm_embeddings(texts, bert_model_path, ctm_tokenizer_path, n_components=12, num_epochs=20, save_path=None):
    # Check if saved embeddings already exist
    if save_path and os.path.exists(save_path):
        print(f"Loading embeddings from {save_path}...")
        embeddings = np.load(save_path)
    else:
        print("Generating BERT+CTM embeddings...")
        bert_ctm_model = BERT_CTM_Model(
            bert_model_path=bert_model_path,
            ctm_tokenizer_path=ctm_tokenizer_path,
            n_components=n_components,
            num_epochs=num_epochs
        )
        embeddings = bert_ctm_model.train(texts)  # Generate embeddings

        # Save embeddings to file
        if save_path:
            print(f"Saving embeddings to file {save_path}...")
            np.save(save_path, embeddings)

    return embeddings


# Data loading and preparation function
def prepare_dataloader(features, labels, batch_size):
    """Create DataLoader for training, validation, and testing"""
    tensor_x = torch.tensor(features, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model training function
def train_model(train_data_path, valid_data_path, test_data_path, train_labels, valid_labels, test_labels,
                bert_model_path, ctm_tokenizer_path, num_heads=8, num_classes=2, epochs=10, batch_size=128,
                learning_rate=5e-3, model_save_path='./final_model.pt'):
    # Step 1: Get BERT+CTM embeddings
    print("Step 1: Getting BERT+CTM embeddings...")
    valid_features = get_bert_ctm_embeddings(valid_data_path, bert_model_path, ctm_tokenizer_path,
                                             save_path='valid_embeddings.npy')
    test_features = get_bert_ctm_embeddings(test_data_path, bert_model_path, ctm_tokenizer_path,
                                            save_path='test_embeddings.npy')
    train_features = get_bert_ctm_embeddings(train_data_path, bert_model_path, ctm_tokenizer_path,
                                             save_path='train_embeddings.npy')

    # Save labels to .npy file
    print("Saving labels to labels.npy file...")
    np.save('train_labels.npy', train_labels)
    np.save('valid_labels.npy', valid_labels)
    np.save('test_labels.npy', test_labels)

    # Step 2: Validate label correctness
    print("Step 2: Validating label correctness...")
    unique_labels_train = np.unique(train_labels)
    unique_labels_valid = np.unique(valid_labels)
    unique_labels_test = np.unique(test_labels)
    print(f"Unique train labels: {unique_labels_train}")
    print(f"Train set class distribution: {np.bincount(train_labels)}")
    print(f"Unique validation labels: {unique_labels_valid}")
    print(f"Validation set class distribution: {np.bincount(valid_labels)}")
    print(f"Unique test labels: {unique_labels_test}")
    print(f"Test set class distribution: {np.bincount(test_labels)}")

    if len(unique_labels_train) != num_classes or len(unique_labels_valid) != num_classes or len(
            unique_labels_test) != num_classes:
        raise ValueError(f"Number of classes in labels does not match expected: expected {num_classes}, "
                         f"but found different classes in training, validation, or test sets")

    # Step 3: Create DataLoader
    print("Step 3: Creating DataLoader...")
    train_loader = prepare_dataloader(train_features, train_labels, batch_size)
    valid_loader = prepare_dataloader(valid_features, valid_labels, batch_size)
    test_loader = prepare_dataloader(test_features, test_labels, batch_size)

    # Step 4: Initialize CNN
    print("Step 4: Initializing CNN...")
    num_filters = 256  # Use 256 convolutional output channels
    kernel_sizes = [2, 3, 4]  # Kernel sizes for convolution
    k = 3 * len(kernel_sizes)
    cnn_output_dim = num_filters * (k + 1)  # Calculate the output feature dimension of CNN

    # Step 5: Initialize attention mechanism
    print("Step 5: Initializing multi-head attention...")
    attention_model = MultiHeadAttentionLayer(embed_size=768, num_heads=8)

    # Step 6: Initialize classifier
    print("Step 6: Initializing classifier...")
    classifier_model = FinalClassifier(input_dim=768, num_classes=num_classes)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 7: Start training
    print("Starting training...")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        classifier_model.train()
        epoch_loss = 0
        y_true = []
        y_pred = []

        # Use tqdm to add progress bar for CNN feature extraction
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            optimizer.zero_grad()
            batch_x = torch.mean(batch_x, dim=1)
            # Extract features from CNN
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x, cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get predicted class
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())

        # Calculate training accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(confusion_matrix(y_true, y_pred))

    # Save model
    torch.save(classifier_model, model_save_path)
    print(f"Trained model has been saved to {model_save_path}")

    # Validation set evaluation
    classifier_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x, cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())

    # Validation accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nValidation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))

    # Test set evaluation
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = extract_CNN_features(batch_x)
            # batch_x = torch.mean(batch_x, dim=1)
            # cnn_output = torch.cat((batch_x, cnn_output), dim=-1)
            attention_output = attention_model(batch_x, batch_x, batch_x)
            outputs = classifier_model(attention_output)
            outputs = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())
    # Test accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\nTest - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # Load and prepare data
    train_data_path = './train.csv'
    valid_data_path = './dev.csv'
    test_data_path = './test.csv'

    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    test_data = pd.read_csv(test_data_path)

    train_labels = train_data['label'].values
    valid_labels = valid_data['label'].values
    test_labels = test_data['label'].values

    # Train model
    bert_model_path = './bert_model'
    ctm_tokenizer_path = './sentence_bert_model'

    # Train model
    train_model(train_data_path, valid_data_path, test_data_path, train_labels, valid_labels, test_labels,
                bert_model_path, ctm_tokenizer_path, num_heads=12, num_classes=2, model_save_path='./final_model.pt')
