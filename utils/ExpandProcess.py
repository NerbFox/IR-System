from .process import process_document, process_single_input, rank_documents_by_similarity, calculate_inverted, process_relevant_documents, process_batch_input_bert, preprocess_data, parse_corpus_file, process_batch_input
from .bert_calculation import get_bert_model_and_tokenizer, compute_bert_document_embeddings, compute_bert, compute_bert_expanded_query
from .bert_calculation import rank_documents_by_similarity as rank_documents_by_similarity_bert, get_document_words
import numpy as np
import os

class ExpandProcess:
    def __init__(self):
        self.source_tf_matrix = None
        self.source_indices = None
        self.vocab = None
        self.input_tf_matrix = None
        self.input_indices = None
        self.tf = None
        self.idf = None
        self.freq = None
        self.relevant = None
        self.ap = []
        self.docs = None
        self.doc_embeddings = None
        self.model = None
        self.tokenizer = None
        self.words_name = None
        self.word_embeddings_name = None
        self.doc_embeddings_name = None
        self.list_ranking = None
        self.list_similarity_scores = None
        self.list_expanded_query = None
        self.query_embeddings = None
        
    def process_source(self, path, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf):
        """
        Process the source documents to create a term frequency matrix.
        
        Args:
            path (str): Path to the source documents.
            stop_word_elim (bool): Whether to eliminate stop words.
            stemming (bool): Whether to apply stemming.
            tf (bool): Whether to compute term frequency.
            idf (bool): Whether to compute inverse document frequency.
            normalize (bool): Whether to normalize the TF-IDF matrix.
            scheme_tf (str): Scheme for computing TF.
            scheme_idf (str): Scheme for computing IDF.
        """
        # Placeholder for actual implementation
        self.source_tf_matrix, self.source_indices, self.vocab, self.docs, self.freq = process_document(
            path, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize
        )
        
        # Get Model and Tokenizer for BERT
        print("Loading BERT model and tokenizer...")
        self.model, self.tokenizer = get_bert_model_and_tokenizer('bert-base-uncased')
        
        # Create Document Embeddings 
        self.doc_embeddings_name = path.split('/')[-1].split('.')[0] + '_bert_document_embeddings.npy'
        print(f"Computing BERT document embeddings and saving to {self.doc_embeddings_name}")
        # print(f"Number of documents: {len(self.docs)}")
        # # type and shape of docs: 
        # print(f"Type of docs: {type(self.docs)}, Shape of docs: {self.docs[0]}")
        # docs is array of tuples (index, content)
        self.docs = list(map(lambda x: x[1], self.docs))  # Extracting only the content from the tuples
        self.doc_embeddings = compute_bert_document_embeddings(
            self.docs, self.model, self.tokenizer, name=self.doc_embeddings_name, recompute=False
        )
        
        # Create Word Embeddings
        self.word_embeddings_name = path.split('/')[-1].split('.')[0] + '_bert_word_embeddings.npy'
        
        self.words_name = path.split('/')[-1].split('.')[0] + '_words.npy'
        words = get_document_words(self.docs, name=self.words_name)
        
        _ = compute_bert_document_embeddings(
            words, self.model, self.tokenizer, name=self.word_embeddings_name, recompute=False
        )
        
        self.tf, self.idf = calculate_inverted(self.freq,scheme_tf=scheme_tf)
        
        print(f"Done processing source documents")
    
    def compute_expanded_query(self, query, k_words=3):
        """
        Compute an expanded query using BERT embeddings.
        
        Args:
            query (str): The original query to expand.
            k_words (int): Number of terms to expand the query with.
        
        Returns:
            tuple: Expanded query terms/words, initial query embeddings.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model and tokenizer are not initialized. Please process source documents first.")
        
        return compute_bert_expanded_query(query, self.docs, self.model, self.tokenizer, k=k_words, name1=self.word_embeddings_name, name2=self.words_name, recompute=False)
    
    def rank_documents_bert(self, query):
        """
        Rank documents based on BERT embeddings similarity.
        
        Args:
            query (str): The query to rank documents against.
            document_embeddings (np.ndarray): Precomputed document embeddings.
        
        Returns:
            tuple: Ranked document indices, similarity scores, and query embedding from BERT.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model and tokenizer are not initialized. Please process source documents first.")
        
        if self.doc_embeddings is None:
            raise ValueError("Document embeddings are not computed. Please process source documents first.")
        
        query_embedding = compute_bert(query, self.model, self.tokenizer).reshape(1, -1)
        ranked_indices, similarity_scores = rank_documents_by_similarity_bert(
            embeddings=self.doc_embeddings,
            query_embedding=query_embedding
        )
        
        return ranked_indices, similarity_scores, query_embedding
            
    def get_ranking(self, index, full_bert):
        """
        Get the ranking of documents for a specific input index.

        Args:
            index (int): The index of the input document.
            full_bert (bool): Whether to use BERT-based ranking.

        Returns:
            list: Ranked document indices and their similarity scores.
        """
        if index not in self.input_indices:
            raise ValueError(f"Index {index} not found in input indices.")

        idx = self.input_indices.index(index)
        if full_bert:
            ranked_results = list(zip(self.list_ranking[idx], self.list_similarity_scores[idx]))
            return sorted(ranked_results, key=lambda x: x[1], reverse=True)

        ranked_results = rank_documents_by_similarity(
            self.input_tf_matrix[idx],
            self.source_tf_matrix,
            self.source_indices
        )
        return ranked_results
    
    def get_inverted(self, index):
        if index in self.source_indices:
            index_position = self.source_indices.index(index)
            return [np.round(self.freq[index_position],5), np.round(self.tf[index_position],5), np.round( self.idf,5)]
        else:
            raise ValueError(f"Index {index} not found in source indices.")
        
    def set_relevant(self, filepath):
        """
        Set the relevant documents for evaluation.
        
        Args:
            relevant (list): List of relevant document indices.
        """
        self.relevant = process_relevant_documents(filepath)
        
    def get_MAP(self, full_bert=True):
        """
        Compute the Mean Average Precision (MAP) for the relevant documents.
        
        Returns:
            float: The computed MAP value.
        """
        if self.relevant is None:
            raise ValueError("Relevant documents not set. Please set relevant documents using set_relevant method.")
        
        average_precisions = []
        
        for input_idx in self.input_indices:
            relevant_docs = set(self.relevant.get(input_idx, []))
            if not relevant_docs:
                continue 
            # print(f"Processing input index: {input_idx}")
            ranked_results = self.get_ranking(input_idx, full_bert=full_bert) 
            ranked_doc_indices = [doc_idx for doc_idx, _ in ranked_results]

            num_relevant = 0
            precision_at_k = []
            for k, doc_idx in enumerate(ranked_doc_indices, 1):
                if doc_idx in relevant_docs:
                    num_relevant += 1
                    precision_at_k.append(num_relevant / k)

            if precision_at_k:
                average_precisions.append(sum(precision_at_k) / len(relevant_docs))
                self.ap.append((input_idx, sum(precision_at_k) / len(relevant_docs)))
            else:
                average_precisions.append(0.0)
                self.ap.append((input_idx, 0.0))

        if not average_precisions:
            return 0.0

        return sum(average_precisions) / len(average_precisions)
    
    def get_ap(self, input_index):
        """
        Get the Average Precision (AP) for a specific input index.
        
        Args:
            input_index (str): The index of the input document.
        
        Returns:
            float: The Average Precision for the specified input index.
        """
        for idx, ap in self.ap:
            if idx == input_index:
                return ap
        return 0.0
    
    def preprocess_and_expand_single(self, input_text, stop_word_elim, stemming, num_of_added):
            docs = [(1, input_text)]  
            queries = preprocess_data(docs, stop_word_elim, stemming)
            # list of tuples: List of (index, preprocessed_content) tuples.
            preprocessed_data_input = [content for _, content in queries]
            # Get the expanded query from tuple (expanded_query, query_embedding)
            # Note: compute_expanded_query returns a tuple (expanded_query, query_embedding)
            expanded_queries = [self.compute_expanded_query(content, k_words=num_of_added)[0] for content in preprocessed_data_input]
            
            preprocessed_data_input = [
            content + ' ' + ' '.join(expanded_query) for content, expanded_query in zip(preprocessed_data_input, expanded_queries)
            ]
            return preprocessed_data_input
    
    def preprocess_and_expand_batch(self, path_to_file, stop_word_elim, stemming, num_of_added):
            docs = parse_corpus_file(path_to_file)
            queries = preprocess_data(docs, stop_word_elim, stemming)
            # list of tuples: List of (index, preprocessed_content) tuples.
            preprocessed_data_input = [content for _, content in queries]
            # Get the expanded query from tuple (expanded_query, query_embedding)
            # Note: compute_expanded_query returns a tuple (expanded_query, query_embedding)
            expanded_results = [self.compute_expanded_query(content, k_words=num_of_added) for content in preprocessed_data_input]
            expanded_queries, initial_query_embeddings = zip(*expanded_results)
            
            preprocessed_data_input = [
            content + ' ' + ' '.join(expanded_query) for content, expanded_query in zip(preprocessed_data_input, expanded_queries)
            ]
            return preprocessed_data_input, initial_query_embeddings
    
    def bert_instant_single(self, input_text, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=1):
        """
        Process a single input text to create a term frequency matrix using BERT.
        
        Args:
            input_text (str): The input text to process.
            stop_word_elim (bool): Whether to eliminate stop words.
            stemming (bool): Whether to apply stemming.
            tf (bool): Whether to compute term frequency.
            idf (bool): Whether to compute inverse document frequency.
            normalize (bool): Whether to normalize the TF-IDF matrix.
            scheme_tf (str): Scheme for computing TF.
            scheme_idf (str): Scheme for computing IDF.
        """
        # Placeholder for actual implementation
        preprocessed_data_input = preprocess_data([(1, input_text)], stop_word_elim, stemming)[0][1]
        # Compute expanded query
        expanded_query, input_query_embedding = self.compute_expanded_query(preprocessed_data_input, k_words=num_of_added)
        preprocessed_data_input += ' ' + ' '.join(expanded_query)
        self.list_expanded_query = [preprocessed_data_input]
        ranked_indices, similarity_scores, expanded_query_embedding = self.rank_documents_bert(
            query=preprocessed_data_input
        )
        
        _, self.input_indices, _ = process_single_input(
            preprocessed_data_input, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
        
        # get source indices from ranked indices
        self.list_ranking = [[self.source_indices[i] for i in ranked_indices]]
        self.list_similarity_scores = [similarity_scores]

        # Write the input_query_embedding and expanded_query_embedding to npy arrays
        os.makedirs('outputs/query_embeddings', exist_ok=True)
        with open(os.path.join('outputs/query_embeddings', 'fullbert_single_input_query_embedding.npy'), 'wb') as f:
            np.save(f, input_query_embedding)
        with open(os.path.join('outputs/query_embeddings', 'fullbert_single_expanded_query_embedding.npy'), 'wb') as f:
            np.save(f, expanded_query_embedding)

        
    def bert_instant_batch(self, path_to_file, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=1):
        """
        Process a batch of input texts to create a term frequency matrix using BERT, returning a ranking.
        
        Args:
            path_to_file (str): Path to the file containing input texts.
            stop_word_elim (bool): Whether to eliminate stop words.
            stemming (bool): Whether to apply stemming.
            tf (bool): Whether to compute term frequency.
            idf (bool): Whether to compute inverse document frequency.
            normalize (bool): Whether to normalize the TF-IDF matrix.
            scheme_tf (str): Scheme for computing TF.
            scheme_idf (str): Scheme for computing IDF.
        """
        # Placeholder for actual implementation
        print(f"Processing instant batch input from {path_to_file}...")
        preprocessed_data_input, initial_query_embeddings = self.preprocess_and_expand_batch(path_to_file, stop_word_elim, stemming, num_of_added)
        
        self.list_expanded_query = preprocessed_data_input
        result = [
            self.rank_documents_bert(
                query=content
            ) for content in preprocessed_data_input
        ]
        
        _, self.input_indices, _ = process_batch_input(
            path_to_file, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
        
        # get source indices from ranked indices
        self.list_ranking = [[self.source_indices[i] for i in ranked_indices] for ranked_indices, _, _ in result]
        self.list_similarity_scores = [list_similarity_scores for _, list_similarity_scores, _ in result]
        expanded_query_embeddings = [query_embedding for _, _, query_embedding in result]

        # Write the initial_query_embeddings and expanded_query_embeddings to npy arrays
        os.makedirs('outputs/query_embeddings', exist_ok=True)
        with open(os.path.join('outputs/query_embeddings', 'fullbert_batch_input_query_embeddings.npy'), 'wb') as f:
            np.save(f, initial_query_embeddings)
        with open(os.path.join('outputs/query_embeddings', 'fullbert_batch_expanded_query_embeddings.npy'), 'wb') as f:
            np.save(f, expanded_query_embeddings)
        
        print(f"Done processing instant batch input from {path_to_file}")
    
    def get_ranking_bert(self, index):
        """
        Get the ranking of documents based on BERT embeddings for a specific input index.
        
        Args:
            index (int): The index of the input document.
        
        Returns:
            list: Ranked document indices and their similarity scores.
        """
        if index in self.input_indices:
            index_position = self.input_indices.index(index)
            return self.list_ranking[index_position], self.list_similarity_scores[index_position]
        else:
            raise ValueError(f"Index {index} not found in input indices.")
    
    def bert_expand_single(self, input_text, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=1):
        """
        Process a single input text to create a term frequency matrix using BERT with expansion.
        
        Args:
            input_text (str): The input text to process.
            stop_word_elim (bool): Whether to eliminate stop words.
            stemming (bool): Whether to apply stemming.
            tf (bool): Whether to compute term frequency.
            idf (bool): Whether to compute inverse document frequency.
            normalize (bool): Whether to normalize the TF-IDF matrix.
            scheme_tf (str): Scheme for computing TF.
            scheme_idf (str): Scheme for computing IDF.
        """
        # Placeholder for actual implementation
        preprocessed_data_input = self.preprocess_and_expand_single(input_text, stop_word_elim, stemming, num_of_added)[0]
        self.list_expanded_query = [preprocessed_data_input]
        self.input_tf_matrix, self.input_indices, _ = process_single_input(
            preprocessed_data_input, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
                
    def bert_expand_batch(self, path_to_file, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf, num_of_added=1):
        """
        Process a single input text to create a term frequency matrix.
        
        Args:
            input_text (str): The input text to process.
            stop_word_elim (bool): Whether to eliminate stop words.
            stemming (bool): Whether to apply stemming.
            tf (bool): Whether to compute term frequency.
            idf (bool): Whether to compute inverse document frequency.
            normalize (bool): Whether to normalize the TF-IDF matrix.
            scheme_tf (str): Scheme for computing TF.
            scheme_idf (str): Scheme for computing IDF.
        """
        # Placeholder for actual implementation
        docs = parse_corpus_file(path_to_file)
        res = preprocess_data(docs, stop_word_elim, stemming)
        res = [
            (id, content + ' ' + ' '.join(self.compute_expanded_query(content, k_words=num_of_added)))
            for id, content in res
        ]
        
        self.list_expanded_query = [content for _, content in res]
        
        self.input_tf_matrix, self.input_indices, _ = process_batch_input_bert(
            res, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )