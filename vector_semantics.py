import numpy as np
import pandas as pd

class TDMatrix:

    def __init__(self, documents, doc_names):
        """ Initialise documents and document names"""
        self.documents = documents
        self.doc_names = doc_names
        self.count = []
        
    
    def term_document_matrix(self):
        """ Create a term document matrix for a set of documents """
        self.vocab = list(set([w for d in self.documents for w in d]))
        
        # count frequency of each word in each document
        for i in range(len(self.documents)):
            self.count.append([self.documents[i].count(w) for w in self.vocab])

        df = pd.DataFrame(self.count, self.doc_names, self.vocab)
        df = df.transpose()
        return df


class TF_IDF_Matrix:

    def __init__(self, documents, doc_names):
        """ Initialise documents and document names"""
        self.documents = documents
        self.doc_names = doc_names
        self.word_freq_count = []
    
    def tf_idf_matrix(self):
        """Create a tf-idf weighted matrix"""
        self.vocab = list(set([w for d in self.documents for w in d]))
        # initialise final matrix with zeros
        self.tf_idf_mat = np.zeros((len(self.vocab),len(self.documents)))

        # cycle through each word in vocabulary to get frequency per document
        for i,w in enumerate(self.vocab):
            self.word_freq_count = np.array([d.count(w) for d in self.documents])
            
            # how many documents each word is in
            self.word_doc_count = 0
            for d in self.documents:
                if w in d:
                    self.word_doc_count+=1

            # calculate tf-idf            
            self.calculate_tf_idf()
            self.tf_idf_mat[i,:] = self.tf_idf

            tf_idf_df = pd.DataFrame(self.tf_idf_mat, self.vocab, self.doc_names)

        return tf_idf_df
    
    def calculate_tf_idf(self):
        """Calculate tf and idf and multiply to get tf-idf weighted values"""
        self.tf = np.log10(self.word_freq_count + 1)
        self.idf = np.log10(len(self.documents)/self.word_doc_count)

        self.tf_idf = self.tf*self.idf
        

class WordMatrix:

    def __init__(self, document, doc_name, window_size):
        """ Initialise documents and document names"""
        self.document = document
        self.document_array = np.array(document)
        self.doc_name = doc_name
        self.vocab = list(set(self.document))
        self.word_word_matrix = np.zeros((len(self.vocab),len(self.vocab)))
        self.window_size = window_size
        
    
    def word_matrix(self):
        """Create a word-word matrix for a single document"""
        # Find the locations of each unique word in the document
        for i in set(self.document):
            word_locs = np.where(self.document_array == i)[0]
            
            # Put all context words for this word into a list
            self.contexts = []
            for j in word_locs:
                self.context = self.document[max(j-self.window_size,0):j] + \
                self.document[j+1:min(j+1+self.window_size,len(self.document))]
                self.contexts += self.context
                
                # Get counts for each context word
                self.get_context_counts()
            
            # Set the context counts vector to the corresponding row in the word-word matrix
            vocab_word_index = np.where(np.array(self.vocab) == i)[0]
            self.word_word_matrix[vocab_word_index,:] = self.context_vector
            
        df_context = pd.DataFrame(self.word_word_matrix, self.vocab, self.vocab)
        return df_context

    def get_context_counts(self):
        """Add counts for each context word to its corresponding position in a context vector"""
        self.context_vector = np.zeros(len(self.vocab))              
        for k in self.contexts:
            word_pos = np.where(np.array(self.vocab) == k)
            self.context_vector[word_pos] = self.contexts.count(k)