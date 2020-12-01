import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
# ML Libraries
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def preprocess_tweet_text(tweet):
    tweet.lower()
    
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    # -------USAR STEMMING O LEMMA A ELECCION (SOLO UNO)----------

    #Stemming
    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]

    #Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(filtered_words)

def get_feature_vector_TFIDF(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"

def get_feautures_BAGWORDS(train_fit):
    matrix = CountVectorizer()
    X = matrix.fit(train_fit)
    return X
    
def main():
    dataset = load_dataset("data/training.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
    # Remover columnas no deseadas
    n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
    #Preprocess data
    dataset.text = dataset['text'].apply(preprocess_tweet_text)
    
    #SELECIIONAR UN METODO DE VETORIZACION (BW O TFIDF)
    # #Extraemos los features de las palabras para usar TFIDF
    # tf_vector = get_feature_vector_TFIDF(np.array(dataset.iloc[:, 1]).ravel())
    # X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
    #Extraemos los features de las palabras para usar Bag of Words
    bw_vector = get_feautures_BAGWORDS(np.array(dataset.iloc[:, 1]).ravel())
    X2 = bw_vector.transform(np.array(dataset.iloc[:,1]).ravel())
    
    #Creamos un array de la columna target (0,4)
    y = np.array(dataset.iloc[:, 0]).ravel()
    
    #Segmentamos nuestros datos para hacer un dataset de entrenamiento y un dataset de test para cada atributo.
    #Un quinto de las muestras será para hacer el test, lo demas para training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    
    # Instanciamos un objeto de training Naive Bayes 
    NB_model = MultinomialNB()
    #Entrenamos el modelo con los datos de entrenamiento
    NB_model.fit(X_train, y_train)
    #Realizamos la prediccion del dataset de test
    y_predict_nb = NB_model.predict(X_test)
    #Calculamos los aciertos en general de nuestro modelo
    print(accuracy_score(y_test, y_predict_nb))
    #Calculamos los aciertos por cada clase de nuestro modelo
    print(classification_report(y_test, y_predict_nb, labels=[0,2,4]))
    


    #-----------------PREDICCION DE TWEETS DE BIDEN--------------------------------------------------


    test_file_name = "trending_tweets/biden_tweets.csv"
    test_ds = load_dataset(test_file_name, ["t_id", "fecha", "user", "text"])
    test_ds = remove_unwanted_cols(test_ds, ["t_id", "fecha", "user"])

    # Creando los features
    test_ds.text = test_ds["text"].apply(preprocess_tweet_text)
    test_feature = bw_vector.transform(np.array(test_ds.iloc[:, 0]).ravel())

    # Prediccion de las clases
    test_prediction_NB = NB_model.predict(test_feature)
    #Imprimimos los tweets y su prediccion
    for i in range(len(test_prediction_NB)):
        print("Tweet=%s, Prediccion=%s" % (test_ds.iloc[i, 0], test_prediction_NB[i]))
   
    



if __name__ == "__main__":
    main()

