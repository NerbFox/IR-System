import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

from collections import defaultdict
import re


from .tf_idf_calculation import compute_tf, compute_idf, compute_tfidf, compute_tfidf_input

nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def parse_corpus_file(filepath):
    """
    Parse a structured corpus file where:
    - .I indicates document ID
    - .T is the title
    - .W is the abstract (content)
    Only the title and abstract are used for retrieval.

    Args:
        filepath (str): Path to the corpus file.

    Returns:
        list of tuples: [(doc_id, full_text), ...]
    """
    documents = []
    doc_id = None
    title = ""
    body = ""
    current_field = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('.I'):
                if doc_id is not None:
                    # Save the previous document before starting a new one
                    full_text = f"{title.strip()} {body.strip()}"
                    documents.append((doc_id, full_text))
                doc_id = int(line.split()[1])
                title = ""
                body = ""
                current_field = None

            elif line == '.T':
                current_field = 'title'
            elif line == '.W':
                current_field = 'body'
            elif line.startswith('.'):
                current_field = None  # Skip other fields like .A, .X
            else:
                if current_field == 'title':
                    title += line + ' '
                elif current_field == 'body':
                    body += line + ' '

        # Add the last document
        if doc_id is not None:
            full_text = f"{title.strip()} {body.strip()}"
            documents.append((doc_id, full_text))

    return documents

def preprocess_data(data, stop_word_elim=False, stemming=False):
    """
    Preprocess the data by tokenizing, removing stop words, and stemming.

    Args:
        data (list of tuples): List of (index, content) tuples to preprocess.
        stop_word_elim (bool): Whether to remove stop words.
        stemming (bool): Whether to apply stemming.

    Returns:
        list of tuples: List of (index, preprocessed_content) tuples.
    """
    preprocessed_data = []

    for idx, doc in data:
        tokens = word_tokenize(doc.lower())

        if stop_word_elim:
            tokens = [word for word in tokens if word not in stop_words]

        if stemming:
            tokens = [stemmer.stem(word) for word in tokens]

        preprocessed_data.append((idx, ' '.join(tokens)))

    return preprocessed_data

def convert_to_tf_dict(indexed_texts):
    """
    Convert a list of (index, content) tuples into a term frequency dict.

    Args:
        indexed_texts (list of tuples): Each tuple is (index, content)

    Returns:
        dict: {doc_id: {term: count}}
    """
    tf_dict = {}

    for doc_id, content in indexed_texts:
        term_freq = defaultdict(int)
        
        # Normalize and tokenize: lowercase and extract words
        tokens = re.findall(r'\b\w+\b', content.lower())
        
        for token in tokens:
            term_freq[token] += 1
        
        tf_dict[doc_id] = dict(term_freq)

    return tf_dict

def convert_to_tf_dict_with_vocab(indexed_texts, vocab):
    """
    Convert a list of (index, content) tuples into a term frequency dict,
    but only count terms that are in the provided vocab. All vocab terms will be present in the dict for each doc, with count 0 if not found.

    Args:
        indexed_texts (list of tuples): Each tuple is (index, content)
        vocab (set or list): Only these words will be counted

    Returns:
        dict: {doc_id: {term: count}} (all terms in vocab, 0 if not present)
    """
    vocab = list(vocab)  # Ensure order is preserved if vocab is a list
    vocab_set = set(vocab)
    tf_dict = {}

    for doc_id, content in indexed_texts:
        term_freq = defaultdict(int)
        # Normalize and tokenize: lowercase and extract words
        tokens = re.findall(r'\b\w+\b', content.lower())
        for token in tokens:
            if token in vocab_set:
                term_freq[token] += 1
        # Ensure all vocab terms are present (with 0 if not found), and order matches vocab
        tf_dict[doc_id] = {term: term_freq.get(term, 0) for term in vocab}

    return tf_dict

def build_tf_matrix(tf_by_doc):
    """
    Build a TF matrix from a dictionary of term frequencies per document.

    Args:
        tf_by_doc (dict): {doc_id: {term: count}}

    Returns:
        tuple: (tf_matrix, doc_indices, vocab)
            tf_matrix (np.ndarray): shape (num_docs, num_terms)
            doc_indices (list): document IDs corresponding to rows
            vocab (list): terms corresponding to columns
    """
    # Step 1: Build vocabulary
    vocab_set = set()
    for tf in tf_by_doc.values():
        vocab_set.update(tf.keys())
    vocab = sorted(vocab_set)
    term_to_index = {term: i for i, term in enumerate(vocab)}

    # Step 2: Build matrix
    doc_indices = sorted(tf_by_doc.keys())
    num_docs = len(doc_indices)
    num_terms = len(vocab)
    tf_matrix = np.zeros((num_docs, num_terms), dtype=np.float32)

    for row_idx, doc_id in enumerate(doc_indices):
        tf_dict = tf_by_doc[doc_id]
        for term, count in tf_dict.items():
            col_idx = term_to_index[term]
            tf_matrix[row_idx, col_idx] = count

    return tf_matrix, doc_indices, vocab

def calculate_weighted(tf_matrix, tf= False, idf= False, scheme_tf="raw", scheme_idf="raw", normalize=False):
    """
    Calculate the weighted TF-IDF matrix.

    Args:
        tf_matrix (np.ndarray): The term frequency matrix.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization to the resulting matrix.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.

    Returns:
        np.ndarray: The weighted TF-IDF matrix.
    """
    
    if tf and idf:
        res = compute_tfidf(tf_matrix, scheme_tf, scheme_idf, normalize)
    elif tf:
        res = compute_tf(tf_matrix, scheme_tf)
    elif idf:
        res = compute_idf(tf_matrix, scheme_idf)
    else:  
        res = tf_matrix

    return res

def calculate_weighted_input(tf_matrix, tf= False, idf= False, scheme_tf="raw", scheme_idf="raw", normalize=False, source_idf=None):
    """
    Calculate the weighted TF-IDF matrix.

    Args:
        tf_matrix (np.ndarray): The term frequency matrix.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization to the resulting matrix.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.
        source_idf (np.ndarray): The IDF vector from the source corpus.

    Returns:
        np.ndarray: The weighted TF-IDF matrix.
    """
    
    if tf and idf:
        res = compute_tfidf_input(tf_matrix, scheme_tf, scheme_idf, normalize, source_idf=source_idf)
    elif tf:
        res = compute_tf(tf_matrix, scheme_tf)
    elif idf:
        res = source_idf
    else:  
        res = tf_matrix

    return res

def calculate_inverted(tf_matrix, scheme_tf="raw", scheme_idf="log"):
    """
    Calculate the TF and IDF matrix from a term frequency matrix for inverted file purposes.

    Args:
        tf_matrix (np.ndarray): The term frequency matrix.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".

    Returns:
        np.ndarray: The TF-IDF matrix.
    """
    
    tf = compute_tf(tf_matrix, scheme=scheme_tf)
    idf = compute_idf(tf_matrix, scheme=scheme_idf)
    
    return tf, idf

def process_document(path_to_document, stop_word_elim=False, stemming=False, tf=False, idf=False, scheme_tf="raw", scheme_idf="raw", normalize=True):
    """
    Process a single document file and return the weighted TF-IDF matrix.

    Args:
        path_to_document (str): Path to the document file.
        stop_word_elim (bool): Whether to remove stop words.
        stemming (bool): Whether to apply stemming.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization.

    Returns:
        np.ndarray: The weighted TF-IDF matrix for the document.
    """
    docs = parse_corpus_file(path_to_document)
    res = preprocess_data(docs, stop_word_elim, stemming)
    tf_dict = convert_to_tf_dict(res)
    tf_matrix, doc_indices, vocab = build_tf_matrix(tf_dict)
    tf_mat_weighted = calculate_weighted(tf_matrix, tf, idf, scheme_tf, scheme_idf, normalize)
    
    return tf_mat_weighted, doc_indices, vocab, docs, tf_matrix

def process_single_input(input_text, vocab_list, stop_word_elim=False, stemming=False, tf=False, idf=False, scheme_tf="raw", scheme_idf="raw", normalize=True, source_idf=None):
    """
    Process a single input text and return the weighted TF-IDF matrix.

    Args:
        input_text (str): The input text to process.
        stop_word_elim (bool): Whether to remove stop words.
        stemming (bool): Whether to apply stemming.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization.
        source_idf (np.ndarray): The IDF vector from the source corpus.

    Returns:
        np.ndarray: The weighted TF-IDF matrix for the input text.
    """
    docs = [(0, input_text)]
    res = preprocess_data(docs, stop_word_elim, stemming)
    tf_dict = convert_to_tf_dict_with_vocab(res, vocab_list)
    tf_matrix, doc_indices, vocab= build_tf_matrix(tf_dict)
    tf_mat_weighted = calculate_weighted_input(tf_matrix, tf, idf, scheme_tf, scheme_idf, normalize, source_idf)
    
    return tf_mat_weighted, doc_indices, vocab

def process_batch_input(path_to_file, vocab_list, stop_word_elim=False, stemming=False, tf=False, idf=False, scheme_tf="raw", scheme_idf="raw", normalize=True, source_idf=None):
    """
    Process a single input text and return the weighted TF-IDF matrix.

    Args:
        input_text (str): The input text to process.
        stop_word_elim (bool): Whether to remove stop words.
        stemming (bool): Whether to apply stemming.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization.
        source_idf (np.ndarray): The IDF vector from the source corpus.

    Returns:
        np.ndarray: The weighted TF-IDF matrix for the input text.
    """
    docs = parse_corpus_file(path_to_file)
    res = preprocess_data(docs, stop_word_elim, stemming)
    tf_dict = convert_to_tf_dict_with_vocab(res, vocab_list)
    tf_matrix, doc_indices, vocab= build_tf_matrix(tf_dict)
    tf_mat_weighted = calculate_weighted_input(tf_matrix, tf, idf, scheme_tf, scheme_idf, normalize, source_idf)
    
    return tf_mat_weighted, doc_indices, vocab

def process_batch_input_bert(res, vocab_list, stop_word_elim=False, stemming=False, tf=False, idf=False, scheme_tf="raw", scheme_idf="raw", normalize=True, source_idf=None, num_of_added = -1):
    """
    Process a single input text and return the weighted TF-IDF matrix.

    Args:
        input_text (str): The input text to process.
        stop_word_elim (bool): Whether to remove stop words.
        stemming (bool): Whether to apply stemming.
        tf (bool): Whether to compute term frequency.
        idf (bool): Whether to compute inverse document frequency.
        scheme_tf (str): TF scheme: "raw", "log", "binary", or "augmented".
        scheme_idf (str): IDF scheme: "raw" or "log".
        normalize (bool): Whether to apply cosine normalization.
        source_idf (np.ndarray): The IDF vector from the source corpus.
        num_of_added (int): Number of added word using bert (-1 = all words)


    Returns:
        np.ndarray: The weighted TF-IDF matrix for the input text.
    """
    tf_dict = convert_to_tf_dict_with_vocab(res, vocab_list)
    tf_matrix, doc_indices, vocab= build_tf_matrix(tf_dict)
    tf_mat_weighted = calculate_weighted_input(tf_matrix, tf, idf, scheme_tf, scheme_idf, normalize, source_idf)
    
    return tf_mat_weighted, doc_indices, vocab

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.
    
    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0
    return dot_product / norm_product

def rank_documents_by_similarity(query_vector, doc_matrix, doc_indices):
    """
    Compute similarity between the query and all documents, return ranked list.

    Args:
        query_vector (np.ndarray): Shape (1, n_features) or (n_features,)
        doc_matrix (list of np.ndarray): List of document vectors (length = num_docs)
        doc_indices (list): List of document IDs corresponding to doc_matrix.

    Returns:
        list of tuples: [(doc_id, similarity), ...] sorted by similarity (descending).
    """
    if isinstance(query_vector, list):
        query_vector = np.array(query_vector[0])  # From process_single_input
    else:
        query_vector = np.array(query_vector)

    similarities = []

    for idx, doc_vector in zip(doc_indices, doc_matrix):
        score = cosine_similarity(query_vector, doc_vector)
        similarities.append((idx, score))

    # Sort by similarity in descending order
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    return ranked

def process_relevant_documents(filepath):
    """
    Reads a qrels.text file and returns a dictionary mapping input index to a list of doc indices.

    Args:
        filepath (str): Path to the qrels.text file.

    Returns:
        dict: {input_index: [doc_index, ...], ...}
    """
    rel = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            x = int(parts[0])
            y = int(parts[1])
            if x not in rel:
                rel[x] = []
            rel[x].append(y)
    return rel
