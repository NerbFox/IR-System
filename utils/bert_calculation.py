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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Computing BERT embeddings on device: {device}", end='\r')
    model = model.to(device)

    encoding = tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True
    )

    # Move tensors to device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

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
    similarities = cosine_similarity(embeddings, query_embedding).flatten()
    
    # Rank documents by similarity
    ranked_indices = np.argsort(similarities)[::-1]  # Descending order
    similarities = similarities[ranked_indices]  # Sort similarities to match ranked_indices
    
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

def compute_bert_document_embeddings(sentences, model, tokenizer, name='bert_document_embeddings.npy', recompute=True):
    """
    Compute BERT embeddings for documents.
    """
    # Check if the embeddings are already saved locally
    name = os.path.join(os.getcwd(), name)
    if os.path.exists(name) and not recompute:
        # print(f"Loading embeddings from {name}")
        embeddings = np.load(name)
    else:
        print(f"Computing and saving embeddings to {name}")
        embeddings = compute_bert(sentences, model, tokenizer)
        np.save(name, embeddings)
    return embeddings

def get_document_words(documents, name='words.npy'):
    """
    Extract unique words from a list of documents.

    Args:
        documents (list): List of documents (strings).

    Returns:
        list: Unique words from the documents.
    """
    # Use absolute path based on current working directory
    name = os.path.join(os.getcwd(), name)
    if os.path.exists(name):
        # print(f"Loading words from {name}", end='\r')
        words = np.load(name, allow_pickle=True).tolist()
    else:
        print(f"Extracting unique words and saving to {name}")
        words = list(dict.fromkeys(
            word for doc in documents for word in doc.split()
        ))
        np.save(name, words)
    return words

def compute_bert_expanded_query(query, documents, model, tokenizer, name1='bert_word_embeddings.npy', name2='words.npy', k=5, recompute=True):
    """
    Compute BERT embeddings for a query and expand it with similar terms.

    Args:
        query (str): The query to expand.
        documents (list): List of documents to use for term expansion.
        model: The BERT model to use for embedding.
        tokenizer: Bert Tokenizer
        k (int): Number of terms to expand the query with.
        recompute (bool): Whether to recompute the embeddings or load from file.
    
    Returns:
        list: The expanded query terms.
    """
    # Compute the query embedding
    query_embedding = compute_bert(query, model, tokenizer).reshape(1, -1)

    # Get unique words from all documents
    document_words = get_document_words(documents, name=name2)
    
    # Compute embeddings for all unique words
    document_embeddings = compute_bert_document_embeddings(document_words, model, tokenizer, name=name1, recompute=recompute)

    # Compute cosine similarity between query and each word embedding
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    # Exclude original query terms from expansion
    query_terms = set(word.lower() for word in query.split())
    # Get indices of top-k most similar words not in the original query
    sorted_indices = np.argsort(similarities)[::-1][:k + len(query_terms)]  
    
    expanded_terms = [
        document_words[idx]
        for idx in sorted_indices
        if document_words[idx].lower() not in query_terms
    ][:k]
    
    return expanded_terms, query_embedding
    

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
    # clean, remove .
    sentences_ = [s.replace('.', '').replace(',', '') for s in sentences]  # Simple cleaning
    time1 = time.time()
    embeddings = compute_bert_document_embeddings(sentences, model, tokenizer, name='bert_document_embeddings.npy', recompute=True)
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
        
    # Example using compute_bert_expanded_query
    expanded_query = compute_bert_expanded_query(query, sentences_, model, tokenizer, k=3, recompute=True, name1='bert_word_embeddings.npy', name2='words.npy')
    print("\nExpanded Query Terms:", expanded_query)
    print("Final Query:", query.split() + expanded_query)
    
    expanded_query_embedding = compute_bert(" ".join(query.split() + expanded_query), model, tokenizer)
    ranked_indices_expanded, sim_expanded = rank_documents_by_similarity(embeddings, expanded_query_embedding)
    print("Similarities with Expanded Query:", sim_expanded)
    print("Ranked Documents with Expanded Query:")
    for idx in ranked_indices_expanded:
        print(sentences[idx])
