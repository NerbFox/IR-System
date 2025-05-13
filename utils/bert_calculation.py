"""Bert calculation module.
This module provides functions for computing BERT embeddings and similarity scores.
It includes:
- BERT embedding calculation
- Cosine similarity calculation
"""

import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def compute_bert(sentences, model, tokenizer):
    """
    Compute BERT embeddings for a list of sentences.

    Args:
        model: The BERT model to use for embedding.
        sentences (list): List of sentences to compute embeddings for.
        tokenizer: Bert Tokenizer

    Returns:
        numpy.ndarray: The computed BERT embeddings.
    """
    if not isinstance(sentences, list):
        sentences = [sentences]  # Ensure sentences is a list
        
    # Encode the sentences
    encoding = tokenizer.batch_encode_plus(
        sentences,
        padding=True,          # Pad to the maximum sequence length
        truncation=True,       # Truncate to the maximum sequence length if necessary
        return_tensors='pt',    # Return PyTorch tensors
        add_special_tokens=True,  # Add special tokens CLS and SEP
        return_attention_mask=True  # Return attention mask
    )
    
    # Compute the embeddings
    with torch.no_grad():
        outputs = model(**encoding) # Use the encoding directly
    
    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Average the token embeddings to get sentence embeddings
    sentence_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
    
    return sentence_embeddings

def rank_documents_by_similarity(embeddings, query_embedding):
    """
    Rank documents based on their cosine similarity to a query embedding.

    Args:
        embeddings (numpy.ndarray): The document embeddings.
        query_embedding (numpy.ndarray): The query embedding.

    Returns:
        tuple:
            numpy.ndarray: Indices of the documents ranked by similarity.
            numpy.ndarray: Cosine similarity scores.
    """
        
    # Compute cosine similarity
    similarities = cosine_similarity(embeddings, query_embedding)
    
    # Rank documents by similarity
    ranked_indices = np.argsort(similarities, axis=0)[::-1]  # Descending order
    ranked_indices = ranked_indices.flatten()  # Flatten to 1D array
    similarities = similarities.flatten()
    
    return ranked_indices, similarities # return similarities

def get_bert_model_and_tokenizer(model_name='bert-base-uncased'):
    """
    Load the BERT model and tokenizer.

    Args:
        model_name (str): The name of the BERT model to load.

    Returns:
        tuple: The loaded BERT model and tokenizer.
    """
    
    folder_name = 'models/'
    # if there is no folder, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    name = os.path.join(folder_name, model_name)
    
    # Check if the model is already saved locally
    if os.path.exists(name):
        # Load the model and tokenizer from local directory
        print(f"Loading model and tokenizer from {name}")
        model = BertModel.from_pretrained(name)
        tokenizer = BertTokenizer.from_pretrained(name)
        return model, tokenizer
    else:
        # Load the model and tokenizer from Hugging Face
        print(f"Downloading and saving model and tokenizer to {name}")
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        # Save the model and tokenizer locally for future use
        model.save_pretrained(name)
        tokenizer.save_pretrained(name)
        return model, tokenizer

def compute_bert_document_embeddings(sentences, model, tokenizer, name='bert_embeddings.npy', recompute=True):
    """
    Compute BERT embeddings for documents.
    """
    # Check if the embeddings are already saved locally
    if os.path.exists(name) and not recompute:
        print(f"Loading embeddings from {name}")
        embeddings = np.load(name)
    else:
        print(f"Computing and saving embeddings to {name}")
        embeddings = compute_bert(sentences, model, tokenizer)
        np.save(name, embeddings)
    return embeddings
    
if __name__ == "__main__":
    # Example usage
    import time
    model_name = 'bert-base-uncased'
    time1 = time.time()
    model, tokenizer = get_bert_model_and_tokenizer(model_name)
    time2 = time.time()
    print("Time taken to download/load model and tokenizer:", time2 - time1)
    
    sentences = [
        "The quick brown fox jumps over the lazy dog. The fox is really quick.",
        "The dog is lazy and sleeps all day. The dog is not quick.",
        "The fox is quick and brown. The fox is very clever too.",
    ]
    
    time1 = time.time()
    embeddings = compute_bert_document_embeddings(sentences, model, tokenizer, name='bert_embeddings.npy', recompute=False)
    time2 = time.time()
    print("Time taken to compute/load embeddings:", time2 - time1)
    
    query = "fox jumps quickly"
    query_embedding = compute_bert(query, model, tokenizer)
    ranked_indices, sim = rank_documents_by_similarity(embeddings, query_embedding)
    
    print("\nDocuments:", sentences)
    print("Similarities:", sim)
    
    print("\nRanked Indices:", ranked_indices)
    print("Query:", query)
    print("Ranked Documents:")
    for idx in ranked_indices:
        print(sentences[idx])
