import os
import os.path
import random
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# Import matplotlib for plotting loss curves
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend, suitable for environments without GUI
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib is not installed. Cannot plot loss curves. You can install it using pip install matplotlib.")


# Global cache manager
class CacheManager:
    def __init__(self, cache_dir="dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache = {}
        self.report_cache = {}
        self.tokenized_cache = {}
        self.max_memory_entries = 5000  # Memory cache maximum entries
        self.stats = {"hit": 0, "miss": 0}
    
    def _get_cache_path(self, key, prefix):
        """Get cache file path"""
        # Use hash to ensure safe and unique file names
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{prefix}_{key_hash}.pkl"
    
    def get_embedding(self, path):
        """Get embedding vector, prioritize from memory cache, then from disk cache, then load original file"""
        if path in self.embedding_cache:
            self.stats["hit"] += 1
            return self.embedding_cache[path]
        
        cache_path = self._get_cache_path(path, "embed")
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Update memory cache
                if len(self.embedding_cache) < self.max_memory_entries:
                    self.embedding_cache[path] = embedding
                
                self.stats["hit"] += 1
                return embedding
            except Exception as e:
                print(f"Error reading embedding cache: {e}")
        
        # Cache miss, load original file
        self.stats["miss"] += 1
        embedding = load_embedding(path)
        
        # Save to disk cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update memory cache
            if len(self.embedding_cache) < self.max_memory_entries:
                self.embedding_cache[path] = embedding
        except Exception as e:
            print(f"Error saving embedding cache: {e}")
        
        return embedding
    
    def get_report(self, path):
        """Get report text, prioritize from memory cache, then from disk cache, then load original file"""
        if path in self.report_cache:
            self.stats["hit"] += 1
            return self.report_cache[path]
        
        cache_path = self._get_cache_path(path, "report")
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    report = pickle.load(f)
                
                # Update memory cache
                if len(self.report_cache) < self.max_memory_entries:
                    self.report_cache[path] = report
                
                self.stats["hit"] += 1
                return report
            except Exception as e:
                print(f"Error reading report cache: {e}")
        
        # Cache miss, load original file
        self.stats["miss"] += 1
        try:
            with open(path, 'r', encoding='utf-8') as f:
                report = f.read()
            
            # Save to disk cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(report, f)
                
                # Update memory cache
                if len(self.report_cache) < self.max_memory_entries:
                    self.report_cache[path] = report
            except Exception as e:
                print(f"Error saving report cache: {e}")
            
            return report
        except Exception:
            return None
    
    def get_tokenized(self, text, tokenizer, max_length=512):
        """Get tokenized text, only use memory cache to avoid serialization issues"""
        # Create a unique key containing text and tokenizer information
        key = f"{tokenizer.__class__.__name__}_{max_length}_{hashlib.md5(text.encode()).hexdigest()}"
        
        if key in self.tokenized_cache:
            self.stats["hit"] += 1
            return self.tokenized_cache[key]
        
        # Cache miss, execute tokenization
        self.stats["miss"] += 1
        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Update memory cache
        if len(self.tokenized_cache) < self.max_memory_entries:
            self.tokenized_cache[key] = tokenized
        
        return tokenized
    
    def clear_memory_cache(self):
        """Clear memory cache but keep disk cache"""
        self.embedding_cache.clear()
        self.report_cache.clear()
        self.tokenized_cache.clear()
    
    # def print_stats(self):
    #     """Print cache hit rate statistics"""
    #     total = self.stats["hit"] + self.stats["miss"]
    #     if total > 0:
    #         hit_rate = (self.stats["hit"] / total) * 100
    #         print(f"Cache hit rate: {hit_rate:.2f}% (hit: {self.stats['hit']}, miss: {self.stats['miss']})")
    #     else:
    #         print("Cache not used yet")

# Create global cache manager instance
cache_manager = CacheManager()


def load_embedding(embedding_path):
    try:
        # Suppress TensorFlow warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        raw_dataset = tf.data.TFRecordDataset([embedding_path])
        embedding_values = None
        
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            embedding_feature = example.features.feature['embedding']
            embedding_values = embedding_feature.float_list.value
        
        if embedding_values is None:
            # Return a zero tensor if no embedding is found
            return torch.zeros(1376, dtype=torch.float32)
            
        return torch.tensor(embedding_values, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading embedding from {embedding_path}: {e}")
        # Return a zero tensor in case of error
        return torch.zeros(1376, dtype=torch.float32)

####Define Dataset Class 
##Here, a list is defined containing a series of diseases (pathologies), which are pathology labels in the dataset.
# These medical imaging labels are typically used for classification tasks (such as diagnosing whether an X-ray shows pneumonia, pleural effusion, etc.).


class MIMIC_Embed_Dataset(Dataset):

    pathologies = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]


#80% training data (train)
#10% validation data (valid)
#10% test data (test)
    split_ratio = [0.8, 0.1, 0.1]

    def __init__(
        self,
        embedpath, # Root directory of embedding .tfrecord files
        csvpath,
        metacsvpath,
        views=["PA"],
        data_aug=None, # Data augmentation (not used currently)
        seed=0, # Set random seed for reproducibility
        unique_patients=True, # Whether to select only one image per patient (to reduce data correlation)
        mode=["train", "valid", "test"][0],
    ):

        super().__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        #Ensures NumPy-based randomness is reproducible.

        self.pathologies = sorted(self.pathologies)

        self.mode = mode
        self.embedpath = embedpath
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(["subject_id", "study_id"])
        self.metacsv = self.metacsv.set_index(["subject_id", "study_id"]) # First read the data, then update its index, logically written in two steps

        self.csv = self.csv.join(self.metacsv).reset_index() # self.csv.join(self.metacsv) means merging (or "concatenating") the columns of self.metacsv into self.csv based on their common indices
        # reset_index() converts the index back to regular columns and generates a new default integer index (0, 1, 2, ...), making it easier to index data by row number in subsequent operations



        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index() # Ensure only one record per patient

        n_row = self.csv.shape[0] # Calculate total number of rows
        
     

        # spit data to one of train valid test
        if self.mode == "train":
            self.csv = self.csv[: int(n_row * self.split_ratio[0])] # Take records from start to 0.8 * n_row as training set
        elif self.mode == "valid":
            self.csv = self.csv[
                int(n_row * self.split_ratio[0]) : int(
                    n_row * (self.split_ratio[0] + self.split_ratio[1])
                )
            ]
        elif self.mode == "test":
            self.csv = self.csv[-int(n_row * self.split_ratio[-1]) :]
        else:
            raise ValueError(
                f"attr:mode has to be one of [train, valid, test] but your input is {self.mode}"
            )

        # Get our classes.
        healthy = self.csv["No Finding"] == 1 
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0 
                mask = self.csv[pathology]

            labels.append(mask.values) # Add this pathology's column data to the labels list
        self.labels = np.asarray(labels).T # .T means transpose, so final self.labels is a matrix of shape (N_samples, 13)
        self.labels = self.labels.astype(np.float32) # Convert label data to float32 type for easier model training and processing

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan  # Replace -1 with NaN so these can be ignored during training

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # add consistent csv values

       
        self.csv["offset_day_int"] = self.csv["StudyDate"] # offset_day_int directly copies the StudyDate field, possibly for later time analysis (e.g., sorting, grouping)

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str) # patientid converts subject_id to string format for path concatenation or image file lookup

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views,
        )

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

    def __len__(self):
        return len(self.labels) # Returns the number of samples, which is the "length" of this dataset - i.e., the number of samples to train

    def __getitem__(self, idx):  # Returns a dictionary
        sample = {}
        sample["idx"] = idx
        sample["lab"] = torch.tensor(self.labels[idx], dtype=torch.float32)  # Convert to tensor

        subjectid = str(self.csv.iloc[idx]["subject_id"]) # Extract patient-related ID from row idx of self.csv
        studyid = str(self.csv.iloc[idx]["study_id"]) 
        dicom_id = str(self.csv.iloc[idx]["dicom_id"]) 


        #data_aug
        embed_file = os.path.join(
            self.embedpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            "s" + studyid,
            dicom_id + ".tfrecord",
        )
        sample["embedding"] = cache_manager.get_embedding(embed_file) 
        #sample["embedding"] = embed_file

        return sample
#     {
#     "idx": 5,
#     "lab": tensor([0., 1., nan, ..., 0.]),
#     "embedding": tensor([...])  # shape: (1376,) or whatever your embedding size is
# }
embedpath = "/home/lde/SPH6004/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files"
csvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-chexpert.csv"
metacsvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-metadata.csv"

# Extended Dataset class that includes reports
class MIMIC_LLM_Dataset(MIMIC_Embed_Dataset):
    def __init__(
        self,
        embedpath,
        csvpath,
        metacsvpath,
        reportpath,
        tokenizer,
        max_report_length=512,
        views=["PA"],
        data_aug=None,
        seed=0,
        unique_patients=True,
        mode=["train", "valid", "test"][0],
    ):
        super().__init__(
            embedpath=embedpath,
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            views=views,
            data_aug=data_aug,
            seed=seed,
            unique_patients=unique_patients,
            mode=mode,
        )
        
        self.reportpath = reportpath
        self.tokenizer = tokenizer
        self.max_report_length = max_report_length
        self.prompt = "analyze this report"

    
    def _load_report(self, subjectid, studyid):
        """Load a report file for a given subject and study ID using cache manager"""
        # Create the report file path
        report_file = os.path.join(
            self.reportpath,
            "p" + subjectid[:2],
            "p" + subjectid,
            f"s{studyid}.txt"
        )
        
        # Default report text if not found
        default_report = self.prompt + ": [No report available]"
        
        # Check if file exists before trying to open it
        if not os.path.exists(report_file):
            return default_report
        
        # Use cache manager to get the report
        report_text = cache_manager.get_report(report_file)
        
        # If report couldn't be loaded or is empty
        if not report_text or len(report_text) < 5:
            return default_report
        
        # Add the prompt to the beginning of the report
        return f"{self.prompt}: {report_text}"
    
    def __getitem__(self, idx):
        # Get the basic sample from the parent class
        try:
            sample = super().__getitem__(idx)
            
            # Add report text
            subjectid = str(self.csv.iloc[idx]["subject_id"])
            studyid = str(self.csv.iloc[idx]["study_id"])
            
            report_text = self._load_report(subjectid, studyid)
            sample["report_text"] = report_text
            
            return sample
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            # Return a fallback sample with dummy data
            return {
                "idx": idx,
                "lab": torch.zeros(len(self.pathologies), dtype=torch.float32),
                "embedding": torch.zeros(1376, dtype=torch.float32),
                "report_text": self.prompt + ": [Error loading data]"
            }


# MLP for processing embeddings
class MLPProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)


# Classification head for the LLM output
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
            # Removed Sigmoid - will be included in BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.layers(x)


# Full model combining LLM, MLP, and classification
class LLMChestXRayClassifier(nn.Module):
    def __init__(
        self, 
        llm_model_name, 
        embedding_dim, 
        mlp_hidden_dim=256, 
        mlp_output_dim=768,
        classifier_hidden_dim=256,
        num_classes=13,
        freeze_llm=True
    ):
        super().__init__()
        
        # Initialize LLM
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.llm_model_name = llm_model_name
        
        # Freeze LLM parameters to save memory and speed up training
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # MLP for processing image embeddings
        self.mlp = MLPProcessor(
            input_dim=embedding_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim
        )
        
        # Get the correct hidden size based on the model
        hidden_size = self.llm.config.hidden_size
        
        # Add projection layer for combining image and text features
        self.projection = nn.Linear(mlp_output_dim + hidden_size, hidden_size)
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=hidden_size,  # Size of the LLM's output
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes
        )
        
        # Add input normalization
        self.input_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, image_embeddings, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            image_embeddings: Tensor of shape (batch_size, embedding_dim)
            input_ids: Tokenized report text, shape (batch_size, seq_length)
            attention_mask: Attention mask for the report text, shape (batch_size, seq_length)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        # Normalize image embeddings
        image_embeddings = self.input_norm(image_embeddings)
        
        # Process image embeddings through MLP
        processed_img_embed = self.mlp(image_embeddings)
        
        # Get LLM embeddings for the text
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract the last hidden states
        last_hidden_states = llm_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Extract the CLS token from the LLM output (first token)
        text_cls_token = last_hidden_states[:, 0, :]
        
        # Combine image and text representations by concatenation
        combined_representation = torch.cat([processed_img_embed, text_cls_token], dim=1)
        
        # Project the combined representation to the expected input size of the classifier
        combined_cls = self.projection(combined_representation)
        
        # Pass through classification head
        logits = self.classifier(combined_cls)
        
        return logits


# Collate function for the DataLoader
def collate_fn(batch):
    """
    Custom collate function to prepare batches for the model
    """
    # Extract components and create lists
    idxs = [item["idx"] for item in batch]
    
    # Ensure labels are tensors
    labels = []
    for item in batch:
        if isinstance(item["lab"], torch.Tensor):
            labels.append(item["lab"])
        else:
            labels.append(torch.tensor(item["lab"], dtype=torch.float32))
    
    # Ensure embeddings are tensors
    embeddings = []
    for item in batch:
        if isinstance(item["embedding"], torch.Tensor):
            embeddings.append(item["embedding"])
        else:
            embeddings.append(torch.tensor(item["embedding"], dtype=torch.float32))
    
    texts = [item["report_text"] for item in batch]
    
    # Stack tensors
    labels = torch.stack(labels)
    embeddings = torch.stack(embeddings)
    
    # Return as a dictionary
    return {
        "idxs": torch.tensor(idxs),
        "labels": labels,
        "embeddings": embeddings,
        "texts": texts
    }


# Training function
def train_model(model, tokenizer, train_loader, val_loader, device, epochs=5, lr=2e-5):
    """
    Train the LLM model
    
    Args:
        model: The LLM model
        tokenizer: Tokenizer for text processing
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train for
        lr: Learning rate
    
    Returns:
        Trained model
    """
    # Move model to device
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss to BCEWithLogitsLoss
    
    best_val_loss = float('inf')
    
    # Store loss values for plotting
    train_losses = []
    val_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    
    print("\nStarting model training:")
    print("=" * 60)
    
    # Print cache usage
    # print("\nInitial cache status:")
    # cache_manager.print_stats()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_losses = []
        
        # Create progress bar for training
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        print("Training Progress:")
        
        # Training loop
        for i, batch in enumerate(train_loader):
            # Move batch to device
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]
            
            # Use cache manager to process text tokenization
            # Tokenize texts in batches to save memory
            encoded_texts_list = []
            for text in texts:
                # Use cache manager to get tokenized text
                encoded_text = cache_manager.get_tokenized(text, tokenizer, max_length=512)
                encoded_texts_list.append(encoded_text)
            
            # Merge encoded texts into batches
            input_ids = torch.cat([encoded_text.input_ids for encoded_text in encoded_texts_list], dim=0).to(device)
            attention_mask = torch.cat([encoded_text.attention_mask for encoded_text in encoded_texts_list], dim=0).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                image_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Handle NaN values in labels
            mask = ~torch.isnan(labels)
            filtered_outputs = outputs[mask]
            filtered_labels = labels[mask]
            
            # Calculate loss
            if filtered_outputs.numel() > 0:
                loss = criterion(filtered_outputs, filtered_labels)
            else:
                # Skip this batch if all values are NaN
                print("Warning: All labels in current batch are NaN, skipping this batch")
                continue
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_value = loss.item()
            train_loss += loss_value
            batch_losses.append(loss_value)
            
            # Print progress
            if (i + 1) % max(1, len(train_loader) // 10) == 0:
                progress = (i + 1) / len(train_loader) * 100
                progress_bar = "=" * int(progress // 2) + ">" + " " * (50 - int(progress // 2))
                print(f"[{progress_bar}] {progress:.1f}% - Batch {i+1}/{len(train_loader)}, Loss: {loss_value:.4f}")
        
        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        batch_val_losses = []
        
        print("\nValidation Progress:")
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # Move batch to device
                embeddings = batch["embeddings"].to(device)
                labels = batch["labels"].to(device)
                texts = batch["texts"]
                
                # Use cache manager to process text tokenization
                encoded_texts_list = []
                for text in texts:
                    # Use cache manager to get tokenized text
                    encoded_text = cache_manager.get_tokenized(text, tokenizer, max_length=512)
                    encoded_texts_list.append(encoded_text)
                
                # Merge encoded texts into batches
                input_ids = torch.cat([encoded_text.input_ids for encoded_text in encoded_texts_list], dim=0).to(device)
                attention_mask = torch.cat([encoded_text.attention_mask for encoded_text in encoded_texts_list], dim=0).to(device)
                
                # Forward pass
                outputs = model(
                    image_embeddings=embeddings,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Handle NaN values in labels
                mask = ~torch.isnan(labels)
                filtered_outputs = outputs[mask]
                filtered_labels = labels[mask]
                
                # Calculate loss
                if filtered_outputs.numel() > 0:
                    loss = criterion(filtered_outputs, filtered_labels)
                else:
                    # Skip this batch if all values are NaN
                    print("Warning: All labels in current batch are NaN, skipping this batch")
                    continue
                loss_value = loss.item()
                val_loss += loss_value
                batch_val_losses.append(loss_value)
                
                # Print progress
                if (i + 1) % max(1, len(val_loader) // 5) == 0:
                    progress = (i + 1) / len(val_loader) * 100
                    progress_bar = "=" * int(progress // 2) + ">" + " " * (50 - int(progress // 2))
                    print(f"[{progress_bar}] {progress:.1f}% - Batch {i+1}/{len(val_loader)}, Loss: {loss_value:.4f}")
        
        # Calculate average validation loss for this epoch
        avg_val_loss = val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)
        
        # Print epoch summary
        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1}/{epochs} Results:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Store all batch losses for this epoch
        train_losses.extend(batch_losses)
        val_losses.extend(batch_val_losses)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"*** Model saved! Current best validation loss: {best_val_loss:.4f} ***")
        print("=" * 60)
    
    # Print final summary
    print("\nTraining completed!")
    print("=" * 60)
    print("Average loss per epoch:")
    for epoch, (train_loss, val_loss) in enumerate(zip(epoch_train_losses, epoch_val_losses)):
        print(f"Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
    print("=" * 60)
    
    # Final cache statistics
    # print("\nFinal cache usage:")
    # cache_manager.print_stats()
    
    # Try to plot losses if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            # Plot epoch losses
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epochs+1), epoch_train_losses, 'b-', label='Training Loss')
            plt.plot(range(1, epochs+1), epoch_val_losses, 'r-', label='Validation Loss')
            plt.title('Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('epoch_losses.png')
            print("Loss curves saved to 'epoch_losses.png'")
            
            # Plot all batch losses
            plt.figure(figsize=(12, 6))
            plt.plot(train_losses, 'b-', alpha=0.5, label='Batch Training Loss')
            plt.plot([len(train_loader)*i for i in range(epochs)], epoch_train_losses, 'bo-', label='Average Training Loss per Epoch')
            plt.title('Loss Changes During Training')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('batch_losses.png')
            print("Batch loss curves saved to 'batch_losses.png'")
            
        except Exception as e:
            print(f"Error plotting loss curves: {e}")
    
    return model


# Evaluation function
def evaluate_model(model, tokenizer, test_loader, device, pathologies=None):
    """
    Evaluate the model on test data
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for text processing
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        pathologies: List of pathology names
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get pathologies if not provided
    if pathologies is None and hasattr(test_loader.dataset, 'pathologies'):
        pathologies = test_loader.dataset.pathologies
    
    all_preds = []
    all_labels = []
    
    print("Testing Progress:")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Move batch to device
            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]
            
            # Use cache manager to process text tokenization
            encoded_texts_list = []
            for text in texts:
                # Use cache manager to get tokenized text
                encoded_text = cache_manager.get_tokenized(text, tokenizer, max_length=512)
                encoded_texts_list.append(encoded_text)
            
            # Merge encoded texts into batches
            input_ids = torch.cat([encoded_text.input_ids for encoded_text in encoded_texts_list], dim=0).to(device)
            attention_mask = torch.cat([encoded_text.attention_mask for encoded_text in encoded_texts_list], dim=0).to(device)
            
            # Forward pass
            outputs = model(
                image_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Apply sigmoid to convert logits to probabilities
            outputs = torch.sigmoid(outputs)
            
            # Store predictions and labels
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Print progress
            if (i + 1) % max(1, len(test_loader) // 5) == 0:
                progress = (i + 1) / len(test_loader) * 100
                progress_bar = "=" * int(progress // 2) + ">" + " " * (50 - int(progress // 2))
                print(f"[{progress_bar}] {progress:.1f}% - Batch {i+1}/{len(test_loader)}")
    
    # Concatenate predictions and labels
    if all_preds and all_labels:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        print("Warning: No evaluable predictions or labels")
        return {"auc": 0, "f1_score": 0, "recall": 0, "precision": 0}
    
    print("\nCalculating evaluation metrics...")
    
    # Calculate metrics
    metrics = {}
    
    # Handle NaN values
    valid_indices = ~np.isnan(all_labels)
    
    # Find optimal thresholds for each class
    optimal_thresholds = []
    print("\nCalculating optimal thresholds for each class:")
    print("-" * 80)
    print(f"{'Class Name':<30} {'Optimal Threshold':<10} {'Positive Ratio':<10} {'F1(0.5 Threshold)':<15} {'F1(Optimal Threshold)':<15}")
    print("-" * 80)
    
    # Per-class metrics with optimal thresholds
    class_metrics = []
    
    from sklearn.metrics import precision_recall_curve
    
    for i in range(all_labels.shape[1]):
        class_result = {}
        mask = valid_indices[:, i]
        
        if mask.sum() > 1:
            y_true = all_labels[mask, i]
            y_pred = all_preds[mask, i]
            
            # Calculate class imbalance ratio
            pos_ratio = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0
            
            # Calculate F1 score with default threshold
            y_pred_binary_default = (y_pred > 0.5).astype(int)
            try:
                f1_default = f1_score(y_true, y_pred_binary_default)
            except:
                f1_default = 0
            
            # Only calculate AUC if we have both classes
            if len(np.unique(y_true)) > 1:
                try:
                    class_result['auc'] = roc_auc_score(y_true, y_pred)
                except Exception as e:
                    class_result['auc'] = 0
                    print(f"Cannot calculate AUC for class {i}: {e}")
            else:
                class_result['auc'] = 0
            
            # Only find optimal threshold if we have positive samples
            if np.sum(y_true) > 0:
                try:
                    # Find optimal threshold using precision-recall curve
                    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                    
                    # Calculate F1 score for each threshold
                    f1_scores = []
                    for p, r in zip(precision[:-1], recall[:-1]):  # Last point has no threshold
                        if p + r > 0:  # Avoid division by zero
                            f1 = 2 * p * r / (p + r)
                        else:
                            f1 = 0
                        f1_scores.append(f1)
                    
                    # Use threshold that gives best F1 score
                    if f1_scores and thresholds.size > 0:
                        best_idx = np.argmax(f1_scores)
                        optimal_threshold = thresholds[best_idx]
                        best_f1 = f1_scores[best_idx]
                    else:
                        # Fallback if we can't compute optimal threshold
                        optimal_threshold = 0.5
                        best_f1 = f1_default
                    
                    # Try different threshold if very small positive ratio
                    # if pos_ratio < 0.05 and optimal_threshold > 0.3:
                    #     # For very imbalanced classes, try a lower threshold
                    #     alternative_threshold = 0.1
                    #     y_pred_binary_alt = (y_pred > alternative_threshold).astype(int)
                    #     f1_alt = f1_score(y_true, y_pred_binary_alt)
                        
                    #     if f1_alt > best_f1:
                    #         optimal_threshold = alternative_threshold
                    #         best_f1 = f1_alt
                    
                    # Apply optimal threshold
                    y_pred_binary = (y_pred > optimal_threshold).astype(int)
                    class_result['f1'] = f1_score(y_true, y_pred_binary)
                    class_result['recall'] = recall_score(y_true, y_pred_binary)
                    class_result['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
                    class_result['threshold'] = optimal_threshold
                    
                    # Print comparison of thresholds
                    path_name = pathologies[i] if i < len(pathologies) else f"Class {i}"
                    print(f"{path_name:<30} {optimal_threshold:<10.3f} {pos_ratio:<10.3f} {f1_default:<15.3f} {class_result['f1']:<15.3f}")
                    
                except Exception as e:
                    class_result['f1'] = class_result['recall'] = class_result['precision'] = 0
                    class_result['threshold'] = 0.5
                    print(f"Cannot calculate optimal threshold for class {i}: {e}")
            else:
                class_result['f1'] = class_result['recall'] = class_result['precision'] = 0
                class_result['threshold'] = 0.5
        else:
            class_result['auc'] = class_result['f1'] = class_result['recall'] = class_result['precision'] = 0
            class_result['threshold'] = 0.5
        
        optimal_thresholds.append(class_result['threshold'])
        class_metrics.append(class_result)
    
    print("-" * 80)
    
    # Calculate and store average metrics
    aucs = [m['auc'] for m in class_metrics if m['auc'] > 0]
    f1s = [m['f1'] for m in class_metrics if m['f1'] > 0]
    recalls = [m['recall'] for m in class_metrics if m['recall'] > 0]
    precisions = [m['precision'] for m in class_metrics if m['precision'] > 0]
    
    metrics['auc'] = np.mean(aucs) if aucs else 0
    metrics['f1_score'] = np.mean(f1s) if f1s else 0
    metrics['recall'] = np.mean(recalls) if recalls else 0
    metrics['precision'] = np.mean(precisions) if precisions else 0
    
    # Print per-class metrics if pathologies are provided
    if pathologies and len(pathologies) == len(class_metrics):
        print("\nUsing optimal thresholds for evaluation metrics:")
        print("-" * 80)
        print(f"{'Class Name':<30} {'Threshold':<8} {'AUC':<8} {'F1 Score':<8} {'Recall':<8} {'Precision':<8}")
        print("-" * 80)
        
        for i, (path, metric) in enumerate(zip(pathologies, class_metrics)):
            print(f"{path:<30} {metric['threshold']:<8.3f} {metric['auc']:<8.3f} {metric['f1']:<8.3f} "
                  f"{metric['recall']:<8.3f} {metric['precision']:<8.3f}")
        
        print("-" * 80)
        print(f"{'Average':<30} {'':<8} {metrics['auc']:<8.3f} {metrics['f1_score']:<8.3f} "
              f"{metrics['recall']:<8.3f} {metrics['precision']:<8.3f}")
        print("-" * 80)
    
    # Try to create a visualization of the results if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            # Create bar chart of F1 scores by pathology
            plt.figure(figsize=(12, 10))
            f1s_by_class = [m['f1'] for m in class_metrics]
            y_pos = np.arange(len(f1s_by_class))
            
            bars = plt.barh(y_pos, f1s_by_class, align='center')
            plt.yticks(y_pos, pathologies)
            plt.xlabel('F1 Score (Using Optimal Threshold)')
            plt.title('F1 Scores by Pathology')
            
            # Add value labels
            for i, v in enumerate(f1s_by_class):
                plt.text(max(0.01, v - 0.1), i, f"{v:.3f}", va='center')
            
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig('f1_by_pathology.png')
            print("F1 Scores by Pathology chart saved to 'f1_by_pathology.png'")

            # Create bar chart of AUC scores by pathology
            plt.figure(figsize=(12, 10))
            aucs_by_class = [m['auc'] for m in class_metrics]
            y_pos = np.arange(len(aucs_by_class))
            
            bars = plt.barh(y_pos, aucs_by_class, align='center')
            plt.yticks(y_pos, pathologies)
            plt.xlabel('AUC Score')
            plt.title('AUC Scores by Pathology')
            
            # Add value labels
            for i, v in enumerate(aucs_by_class):
                plt.text(max(0.01, v - 0.1), i, f"{v:.3f}", va='center')
            
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig('auc_by_pathology.png')
            print("AUC Scores by Pathology chart saved to 'auc_by_pathology.png'")
            
            # Create new visualization for positive rate vs metrics
            # Create two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot AUC vs Positive Rate
            aucs = [m['auc'] for m in class_metrics]
            ax1.scatter(pos_ratios, aucs, s=100, alpha=0.7)
            for i, (x, y) in enumerate(zip(pos_ratios, aucs)):
                ax1.annotate(pathologies[i], (x, y),
                           xytext=(5, 5), textcoords='offset points')
            ax1.set_xlabel('Positive Rate')
            ax1.set_ylabel('AUC')
            ax1.set_title('Positive Rate vs AUC')
            ax1.grid(True)
            
            # Plot F1 Score vs Positive Rate
            f1s = [m['f1'] for m in class_metrics]
            ax2.scatter(pos_ratios, f1s, s=100, alpha=0.7)
            for i, (x, y) in enumerate(zip(pos_ratios, f1s)):
                ax2.annotate(pathologies[i], (x, y),
                           xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('Positive Rate')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Positive Rate vs F1 Score')
            ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('metrics_vs_positive_rate.png')
            print("Metrics vs Positive Rate chart saved to 'metrics_vs_positive_rate.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")

    # Show predictions for first 10 test samples
    print("\nPredictions for first 10 test samples:")
    print("-" * 100)
    print(f"{'Pathology':<30} {'True Label':<12} {'Predicted':<12} {'Probability':<12}")
    print("-" * 100)

    # Get first 10 predictions and labels
    first_10_preds = all_preds[:10]
    first_10_labels = all_labels[:10]

    for i in range(10):  # For each sample
        print(f"\nSample {i+1}:")
        print("-" * 100)
        for j, pathology in enumerate(pathologies):  # For each pathology
            true_label = first_10_labels[i][j]
            pred_prob = first_10_preds[i][j]
            pred_label = 1 if pred_prob > optimal_thresholds[j] else 0
            
            # Skip if true label is NaN
            if np.isnan(true_label):
                continue
                
            print(f"{pathology:<30} {true_label:<12.0f} {pred_label:<12.0f} {pred_prob:<12.3f}")
    print("-" * 100)

    return metrics, class_metrics, optimal_thresholds


# Main function to run the entire pipeline
def main(use_cache=True, cache_dir="dataset_cache", scheduler_type="plateau"):
    """
    Main function to run the entire training and evaluation pipeline
    
    Args:
        use_cache: Whether to use the cache system
        cache_dir: Cache directory path
        scheduler_type: Learning rate scheduler type ("plateau", "cosine", "none")
    """
    # Set up cache directory
    global cache_manager
    if use_cache:
        cache_manager = CacheManager(cache_dir=cache_dir)
        print(f"Data cache enabled, cache directory: {cache_dir}")
    else:
        # Create an empty cache manager whose methods will directly call original functions
        print("Data cache disabled")
        class DummyCacheManager:
            def get_embedding(self, path):
                return load_embedding(path)
            
            def get_report(self, path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    return None
            
            def get_tokenized(self, text, tokenizer, max_length=512):
                return tokenizer(
                    text, 
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
            
            # def print_stats(self):
            #     print("Cache is disabled")
                
            def clear_memory_cache(self):
                pass
                
        cache_manager = DummyCacheManager()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Suppress TensorFlow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths
    embedpath = "/home/lde/SPH6004/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files"
    csvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-chexpert.csv"
    metacsvpath = "/home/lde/SPH6004/mimic-cxr-2.0.0-metadata.csv"
    reportpath = "/home/lde/SPH6004/mimic-cxr-reports/files"
    
    # Define model parameters
    llm_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    embedding_dim = 1376  # Dimension of the image embeddings
    
    # Use smaller batch size to reduce memory usage
    batch_size = 32
    
    print("\n" + "="*60)
    print("Chest X-Ray LLM Classification Task")
    print("="*60)
    print(f"Model: {llm_model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Embedding Dimension: {embedding_dim}")
    print(f"Cache Status: {'Enabled' if use_cache else 'Disabled'}")
    print("="*60)
    
    # Initialize tokenizer
    try:
        print("Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        print(f"✓ Successfully loaded tokenizer: {llm_model_name}")
    except Exception as e:
        print(f"× Error loading tokenizer: {e}")
        print("Falling back to default BERT tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    try:
        # Create datasets
        print("\nCreating datasets...")
        train_dataset = MIMIC_LLM_Dataset(
            embedpath=embedpath,
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            reportpath=reportpath,
            tokenizer=tokenizer,
            mode="train"
        )
        
        val_dataset = MIMIC_LLM_Dataset(
            embedpath=embedpath,
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            reportpath=reportpath,
            tokenizer=tokenizer,
            mode="valid"
        )
        
        test_dataset = MIMIC_LLM_Dataset(
            embedpath=embedpath,
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            reportpath=reportpath,
            tokenizer=tokenizer,
            mode="test"
        )
        
        print(f"✓ Successfully created datasets - Training: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Print pathology classes
        print("\nTarget Pathology Classes:")
        for i, pathology in enumerate(train_dataset.pathologies):
            print(f"  {i+1}. {pathology}")
        
        # Create data loaders with smaller batch size
        print("\nCreating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # No multiprocessing to avoid errors
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        print("✓ Successfully created data loaders")
        
        # Create model
        print(f"\nCreating model...")
        model = LLMChestXRayClassifier(
            llm_model_name=llm_model_name,
            embedding_dim=embedding_dim,
            num_classes=len(train_dataset.pathologies)
        )
        print(f"✓ Successfully created model {llm_model_name}")
        
        # Print model structure summary
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal model parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/num_params*100:.2f}%)")
        
        # Train model
        print("\nStarting model training...")
        model = train_model(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=5
        )
        
        # Load the best model before evaluation
        print("\nLoading best model parameters...")
        try:
            best_model_state = torch.load("best_model.pt")
            model.load_state_dict(best_model_state)
            print("✓ Successfully loaded best model parameters")
        except Exception as e:
            print(f"× Error loading best model parameters: {e}")
            print("Will use the last epoch's model parameters for evaluation")
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics, class_metrics, optimal_thresholds = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_loader=test_loader,
            device=device,
            pathologies=train_dataset.pathologies
        )
        
        # Print metrics
        print("\nTest Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-" * 30)
        
        # Clear cache to save memory
        if use_cache:
            print("\nClearing memory cache...")
            cache_manager.clear_memory_cache()
    
    except Exception as e:
        print(f"\n× Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Default to enabled cache
    main(use_cache=True)