import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords #imports the downloaded stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

def main():
    st.title("Restaurant Review Classification Web App")
    st.sidebar.title("Restaurant Review Classification Web App")
    st.markdown("Let's classify the reviews")
    st.sidebar.markdown("Let's classify the reviews")

    @st.cache(persist=True)
    def load_data():
        dataset = pd.read_csv("C:\datasets\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
        return dataset
    
    def clean_data(df):
        corpus = [] #all reviews after being cleaned
        for i in range(0, 1000):
            review = re.sub('[^a-zA-Z]', ' ',df['Review'][i]) #replaces any element thats not a letter gets replaced by space
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stop_words = stopwords.words('english')
            all_stop_words.remove("not")
            review = [ps.stem(word) for word in review if not word in set(all_stop_words)]
            review = ' '.join(review)
            corpus.append(review)
        return corpus
   
    
    def split(cf):
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(cf).toarray()
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        return X_train, X_test, y_train, y_test
    
        
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
    
 
    class_names = ['Good Review', 'Bad Review']
    df = load_data()
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("raw data before cleaning")
        st.write(df)
        st.markdown("This dataset consists of 1000 restaurant reviews")
    
    cf = clean_data(df)
    if st.sidebar.checkbox("Show cleaned data", False,key="clean_data"):
        st.subheader("Cleaned data for better classification")
        st.write(cf)
        st.markdown("From the set of 1000 reviews this dataset consists of all the clean reviews which are free of punctuation,stopwords and consists of stem words for better classification")
    
    X_train, X_test, y_train, y_test = split(cf)


    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Naive Bays", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Naive Bays':
        # st.sidebar.subheader("Model Hyperparameters")
        # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        # max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = GaussianNB()
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        criterion = st.sidebar.radio("Choose the criterion",("gini","entropy"), key='criterion')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

  
  



if __name__ == '__main__':
    main()
