from .process import process_document, process_single_input, rank_documents_by_similarity, calculate_inverted, process_batch_input, process_relevant_documents
import numpy as np

class OriginalProcess:
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
        self.source_tf_matrix, self.source_indices, self.vocab, _ = process_document(
            path, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize
        )
        
        self.freq, self.tf, self.idf = calculate_inverted(self.source_tf_matrix,scheme_tf=scheme_tf)
                
    def process_single_input(self, input_text, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf):
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
        self.input_tf_matrix, self.input_indices, _ = process_single_input(
            input_text, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
        
        
    def process_batch_input(self, path_to_file, stop_word_elim, stemming, tf, idf, normalize, scheme_tf, scheme_idf):
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
        self.input_tf_matrix, self.input_indices, _ = process_batch_input(
            path_to_file, self.vocab, stop_word_elim, stemming, tf, idf, scheme_tf, scheme_idf, normalize, source_idf=self.idf
        )
        
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
        