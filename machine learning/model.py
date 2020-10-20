import numpy as np 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from processing import Tf_idf, CountVec






class Model():
    def __init__(self,model="LogisticRegression",preproc = "TF_IDF",parameters={}):
        self.parameters = parameters
        self.model = model
        self.preproc = preproc
        self._fit = False

    # Model loss
    def _multiclass_logloss(self,actual, predicted, eps=1e-15):
        """Multi class version of Logarithmic Loss metric.
        :param actual: Array containing the actual target classes
        :param predicted: Matrix with class predictions, one probability per class
        """
        # Convert 'actual' to a binary array if it's not already:
        if len(actual.shape) == 1:
            actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
            for i, val in enumerate(actual):
                actual2[i, val] = 1
            actual = actual2

        clip = np.clip(predicted, eps, 1 - eps)
        rows = actual.shape[0]
        vsota = np.sum(actual * np.log(clip))
        return -1.0 / rows * vsota

    def graph(self,display_matrix=False):
        x=self._pipeline.fit(self.xvalid)
        y=self.yvalid
            # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                        ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(self.clf, x,y ,
                                        display_labels=self.lbl_enc.classes_,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)
            tick_marks = np.arange(len(self.lbl_enc.classes_))
            plt.xticks(tick_marks, self.lbl_enc.classes_, rotation=90)
            if display_matrix == True :
                print(title)
                print(disp.confusion_matrix)

        plt.show()

    def _split(self,ratio):
        self.df = self.df.groupby('y')
        self.df = self.df.filter(lambda labels: len(labels) > 1)
        self.lbl_enc = preprocessing.LabelEncoder()
        self.y = self.lbl_enc.fit_transform(self.df.y.values)
        self.xtrain, self.xvalid, self.ytrain, self.yvalid = train_test_split(self.df.X.values, self.y, 
                                                  stratify=self.y, 
                                                  random_state=42, 
                                                  test_size=ratio,shuffle=True)
        self._lengthX_train = len(self.xtrain)
        self._lengthX_valid = len(self.xvalid)


    def _preproc(self):
        if self.preproc == "TF_IDF" : 
            self._pipeline = Tf_idf(self.xtrain)

        elif self.preproc == "CountVec" : 
            self._pipeline = CountVec(self.xtrain)
            
        else :
            raise ValueError("This process doesn't exist")

        self.xtrain= self._pipeline.fit(self.xtrain)
            

    def _fit_model(self):
        if self.model == "LogisticRegression" :
            self.clf = LogisticRegression(**self.parameters)

        elif self.model == "MultinomialNB" :
            self.clf = MultinomialNB(**self.parameters)
        
        elif self.model == 'SVM' :
            self.clf = SVC(**self.parameters)

        else :
            raise ValueError("This model doesn't exist")
        
        self.clf.fit(self.xtrain, self.ytrain)

        
    
    def pred(self,x):
        if len(x) == 0 :
            raise ValueError("X is empty")
        if self._fit == True :
            x = self._pipeline.fit(x)
            return self.clf.predict(x)
        print("You need to fit your model with data first")

    def _eval(self):
        predictions = self.pred(self.xvalid)
        self._acc = accuracy_score(predictions, self.yvalid)
        print("accuracy : ",self._acc )

    def details(self):
        print("Model : ",self.model)
        print("Pre-processing : ",self.preproc)
        if self._fit == True :
            print("length trainning : ",self._lengthX_train)
            print("length evluation : ",self._lengthX_valid)
            print("Accuracy : ",self._acc)

    def fit(self,df,ratio=0.3):
        self.df = df 
        
        self._split(ratio)
        self._preproc()
        self._fit_model()

        self._fit = True
        
        self._eval()

        






        
    

        

