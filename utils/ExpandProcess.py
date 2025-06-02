from .process import process_document, process_single_input, rank_documents_by_similarity, calculate_inverted, process_relevant_documents, process_batch_input_bert, preprocess_data, parse_corpus_file
from .bert_calculation import get_bert_model_and_tokenizer, compute_bert_document_embeddings, compute_bert, compute_bert_expanded_query
from .bert_calculation import rank_documents_by_similarity as rank_documents_by_similarity_bert
import numpy as np

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
        self.word_embeddings = None
        self.model = None
        self.tokenizer = None
        self.word_embeddings_name = None
        self.doc_embeddings_name = None
        self.ranking = None
        self.list_ranking = None
        self.similarity_scores = None
        self.list_similarity_scores = None
        
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
        self.source_tf_matrix, self.source_indices, self.vocab, self.docs = process_document(
            path, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize
        )
        
        # Get Model and Tokenizer for BERT
        print("Loading BERT model and tokenizer...")
        self.model, self.tokenizer = get_bert_model_and_tokenizer('bert-base-uncased')
        
        # Create Document Embeddings 
        self.doc_embeddings_name = path.split('/')[-1].split('.')[0] + '_bert_document_embeddings.npy'
        print(f"Computing BERT document embeddings and saving to {self.doc_embeddings_name}")
        self.doc_embeddings = compute_bert_document_embeddings(
            self.docs, self.model, self.tokenizer, name=self.doc_embeddings_name, recompute=False
        )
        
        # Create Word Embeddings
        self.word_embeddings_name = path.split('/')[-1].split('.')[0] + '_bert_word_embeddings.npy'
        print(f"Computing BERT word embeddings and saving to {self.word_embeddings_name}")
        self.word_embeddings = compute_bert_document_embeddings(
            self.vocab, self.model, self.tokenizer, name=self.word_embeddings_name, recompute=False
        )
        
        self.freq, self.tf, self.idf = calculate_inverted(self.source_tf_matrix,scheme_tf=scheme_tf)
    
    def compute_expanded_query(self, query, k_words=3):
        """
        Compute an expanded query using BERT embeddings.
        
        Args:
            query (str): The original query to expand.
            k_words (int): Number of terms to expand the query with.
        
        Returns:
            list: Expanded query terms/words.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model and tokenizer are not initialized. Please process source documents first.")
        
        return compute_bert_expanded_query(query, self.docs, self.model, self.tokenizer, k=k_words, name=self.word_embeddings_name, recompute=False)
    
    def rank_documents_bert(self, query, k_words=3):
        """
        Rank documents based on BERT embeddings similarity.
        
        Args:
            query (str): The query to rank documents against.
            document_embeddings (np.ndarray): Precomputed document embeddings.
        
        Returns:
            list: Ranked document indices and their similarity scores.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model and tokenizer are not initialized. Please process source documents first.")
        
        if self.doc_embeddings is None:
            raise ValueError("Document embeddings are not computed. Please process source documents first.")
        
        expanded_query = self.compute_expanded_query(query, k_words)
        query_final = ' '.join(expanded_query)
        query_embedding = compute_bert(query_final, self.model, self.tokenizer).reshape(1, -1)
        ranked_indices, similarity_scores = rank_documents_by_similarity_bert(
            embeddings=self.doc_embeddings,
            query_embedding=query_embedding
        )
        
        return ranked_indices, similarity_scores
            
    def get_ranking(self, index):
        if index in self.input_indices:
            index_position = self.input_indices.index(index)
            ranked_results = rank_documents_by_similarity(
                self.input_tf_matrix[index_position],
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
        
    def get_MAP(self):
        """
        Compute the Mean Average Precision (MAP) for the relevant documents.
        
        Returns:
            float: The computed MAP value.
        """
        if self.relevant is None or self.input_indices is None:
            raise ValueError("Relevant documents not set. Please set relevant documents using set_relevant method.")
        
        average_precisions = []
        
        for input_idx in self.input_indices:
            relevant_docs = set(self.relevant.get(input_idx, []))
            if not relevant_docs:
                continue 

            ranked_results = self.get_ranking(input_idx) 
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
    
    def preprocess_and_expand_batch(self, path_to_file, stop_word_elim, stemming, num_of_added):
            docs = parse_corpus_file(path_to_file)
            queries = preprocess_data(docs, stop_word_elim, stemming)
            # list of tuples: List of (index, preprocessed_content) tuples.
            preprocessed_data_input = [content for _, content in queries]
            expanded_queries = [self.compute_expanded_query(content, k_words=num_of_added) for content in preprocessed_data_input]
            
            preprocessed_data_input = [
            content + ' ' + ' '.join(expanded_query) for content, expanded_query in zip(preprocessed_data_input, expanded_queries)
            ]
            return preprocessed_data_input
    
    def bert_instant_single(self, input_text, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=-1):
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
        preprocessed_data_input = self.preprocess_and_expand_batch([input_text], stop_word_elim, stemming, num_of_added)[0]
        
        ranked_indices, similarity_scores = self.rank_documents_bert(
            embeddings=self.doc_embeddings,
            query_embedding=compute_bert(str(preprocessed_data_input), self.model, self.tokenizer).reshape(1, -1)
        )
        
        # get source indices from ranked indices
        self.ranking = [self.source_indices[i] for i in ranked_indices]
        self.similarity_scores = similarity_scores
        
    def bert_instant_batch(self, path_to_file, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=-1):
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
        preprocessed_data_input = self.preprocess_and_expand_batch(path_to_file, stop_word_elim, stemming, num_of_added)
        
        result = [
            self.rank_documents_bert(
                embeddings=self.doc_embeddings,
                query_embedding=compute_bert(str(content), self.model, self.tokenizer).reshape(1, -1)
            ) for content in preprocessed_data_input
        ]
        # get source indices from ranked indices
        self.list_ranking = [[self.source_indices[i] for i in ranked_indices] for ranked_indices, _ in result]
        self.list_similarity_scores = [list_similarity_scores for _, list_similarity_scores in result]
    
    def bert_expand_single(self, input_text, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf="log", num_of_added=-1):
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
        preprocessed_data_input = self.preprocess_and_expand_batch([input_text], stop_word_elim, stemming, num_of_added)[0]
        
        self.input_tf_matrix, self.input_indices, _ = process_single_input(
            preprocessed_data_input, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
                
    def bert_expand_batch(self, path_to_file, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf, num_of_added=-1):
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
        
        self.input_tf_matrix, self.input_indices, _ = process_batch_input_bert(
            res, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )