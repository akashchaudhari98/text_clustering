from pickle import TRUE
from sentence_transformers import SentenceTransformer
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import time
import matplotlib.pyplot as plt
import hdbscan
import umap
import pandas as pd
from sklearn.cluster import KMeans


class clustering:
    def __init__(self,
                 file_location,
                 text_file=True,
                 csv_file=False,
                 column_name= "") -> None:

        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.data = file_location
        self.nlp = spacy.load("en_core_web_sm")

        if text_file:
            if file_location == None:
                raise ValueError("please pass data file location")
            with open(self.data, errors="ignore") as file_:
                self.data = file_.read()
            self.divided_data = self.para_div(self.data)
            
            print("no of paras {} " .format(len(self.divided_data)))

        if csv_file:
            self.column = column_name
            if file_location == None:
                raise ValueError("please pass data file location")
            self.divided_data = pd.read_csv(self.data, sep='\t')
            self.divided_data = self.divided_data[self.column]
            if len(self.divided_data) > 1000:
                self.divided_data = self.divided_data[0:10000]
            print("no of paras {} ".format(len(self.divided_data)))

        self.cleaned_data = self.cleaning(self.divided_data)
        self.encoded_data = self.sentence_encode(self.cleaned_data)
        self.reduced_data = self._umap(data=self.encoded_data)
        self.clusters = self._dbscan(self.reduced_data)

        self.umap_data = self._umap(
            n_neighbors=2, data=self.encoded_data, min_dist=0.0, n_components=2,)
        
        result = pd.DataFrame(self.umap_data, columns=['x', 'y'])
        result['labels'] = self.clusters.labels_

        # Visualize clusters
        fig, ax = plt.subplots(figsize=(10, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.5)
        plt.scatter(clustered.x, clustered.y,
                    c=clustered.labels, s=0.5, cmap='hsv_r')
        plt.colorbar()
        plt.savefig('hdbscan.png')

        print("plotted")

    def cleaning(self, data_list):

        clean_data = []
        for doc in data_list:
            doc = str(doc)
            doc = re.sub(r'\W+', ' ', doc)
            doc = self.nlp(doc)
            doc = " ".join([str(token) for token in doc if token.pos_ != "ADP" and
                            token.pos_ != "PROPN" and
                            # token.pos_ != "PRON" and
                            token.is_stop != True and
                            token.pos_ != "NUM" and
                            # token.pos_ != "VERB" and
                            token.pos_ != "DET"])
            clean_data.append(doc)

        return clean_data

    def para_div(self, data):
        ''' divide the book into paragraphs'''
        word_list = word_tokenize(data)
        count = 0
        data_list = []
        temp_list = []
        for words in word_list:
            if count != 200:
                temp_list.append(words)
                count = count + 1
            else:
                temp_list = " ".join(temp_list)
                data_list.append(temp_list)
                count = 0
                temp_list = []
        print("size of para 1 ", len(data_list[0].split(" ")))
        print("total no of para ", len(data_list))

        return data_list

    def sentence_encode(self, data):
        ''' encode sentences using sentence transofrmers'''

        embeddings = self.model.encode(data, show_progress_bar=True)

        return embeddings

    def _umap(self, data, n_neighbors=15, n_components=5, metric='cosine', min_dist=0.1):
        # def _umap(self,**kwargs):
        ''' dimentionality reduction'''

        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components,
                                    metric=metric, min_dist=min_dist).fit_transform(data)
        return umap_embeddings

    def _dbscan(self, data):
        ''' create clusters'''
        cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  metric='euclidean',
                                  cluster_selection_method='eom').fit(data)

        return cluster


if __name__ == "__main__":
    book = "C:/Users/akash/Desktop/The Da Vinci Code.txt"
    csv_location = 'D:/projects/train.tsv'
    clustering(file_location=csv_location,
               text_file=False,
               csv_file=True,
               column_name =  "Phrase")
