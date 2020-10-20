
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords


stopwords_list = stopwords.words('french')
stopwords_list = None

class Tf_idf():
    def __init__(self,xtrain):
        # Always start with these features. They work (almost) everytime!
        self.tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = stopwords_list)

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        self.tfv.fit(list(xtrain))

    def fit(self,x):
        return self.tfv.transform(x)


class CountVec():
    def __init__(self,xtrain):
        self.ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = stopwords_list)

        self.ctv.fit(list(xtrain))

    def fit(self,x):
        return self.ctv.transform(x)