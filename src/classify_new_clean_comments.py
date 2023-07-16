
import openpyxl
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import nltk
import string                              # for string operations
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer  
from nltk.corpus import stopwords
import spacy
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os

#Modeling Phase

df = pd.read_pickle('../resources/common/data/final_cleaned_Comments.pkl')
df_labled=pd.read_pickle('../resources/common/data/final_cleaned_labled_Comments.pkl')


# # Semi-supervised Learning with LSTM

# In[112]:


np.random.seed(42)

# Étape 1: Entraînez un modèle sur les données étiquetées

small_texts = df_labled['lemmatized_tokens_finale'].values
small_labels = df_labled['score'].values

# Encodage des étiquettes
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(small_labels)
small_y = to_categorical(encoded_labels)

# Tokenization et padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(small_texts)
sequences = tokenizer.texts_to_sequences(small_texts)
word_index = tokenizer.word_index
small_X = pad_sequences(sequences)

# Construction du modèle
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=small_X.shape[1]))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(small_y.shape[1], activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(small_X, small_y, epochs=12, batch_size=32)

# Étape 2: Utilisez le modèle pour prédire des étiquettes pour les données non étiquetées

# Supposons que df soit votre DataFrame contenant les données non étiquetées
# et que la colonne 'lemmatized_tokens' contient les textes déjà nettoyés, tokenisés et lemmatisés

# Recombiner les tokens en texte
unlabeled_texts = [' '.join(tokens) for tokens in df['lemmatized_tokens_finale'].values]
unlabeled_sequences = tokenizer.texts_to_sequences(unlabeled_texts)
unlabeled_X = pad_sequences(unlabeled_sequences, maxlen=small_X.shape[1])

# Prédiction
predictions = model.predict(unlabeled_X)
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Étape 3: Combinez les données étiquetées originales avec les données auto-étiquetées
combined_texts = np.concatenate((small_texts, unlabeled_texts))
combined_labels = np.concatenate((small_labels, predicted_labels))

# Encodage des étiquettes combinées
encoded_combined_labels = label_encoder.transform(combined_labels)
combined_y = to_categorical(encoded_combined_labels)

# Tokenization et padding des textes combinés
combined_sequences = tokenizer.texts_to_sequences(combined_texts)
combined_X = pad_sequences(combined_sequences, maxlen=small_X.shape[1])

# Étape 4: Ré-entraînez le modèle sur le nouvel ensemble de données combiné
model.fit(combined_X, combined_y, epochs=10, batch_size=32,shuffle=True)
# Préparer les données de la grande dataset pour la prédiction
# Comme précédemment mentionné, 'lemmatized_tokens' contient les textes déjà nettoyés, tokenisés et lemmatisés
large_unlabeled_texts = [' '.join(tokens) for tokens in df['lemmatized_tokens_finale'].values]
large_unlabeled_sequences = tokenizer.texts_to_sequences(large_unlabeled_texts)
large_unlabeled_X = pad_sequences(large_unlabeled_sequences, maxlen=small_X.shape[1])

# Utiliser le modèle pour prédire les étiquettes de la grande dataset
large_predictions = model.predict(large_unlabeled_X)

# Convertir les prédictions en étiquettes lisibles
large_predicted_labels = np.argmax(large_predictions, axis=1)
large_predicted_labels = label_encoder.inverse_transform(large_predicted_labels)

# Ajouter les étiquettes prédites à la grande dataset
df = df.copy()
df['predicted_score_lstm'] = large_predicted_labels


# In[113]:


df['predicted_score_lstm']


# # Semi-supervised Learning with TfidfVectorizer and Naive Bayes

# In[114]:


# # Préparation des données étiquetées
# df_labled = df_labled.copy()
# df_labled['lemmatized_tokens_finale'] = df_labled['lemmatized_tokens_finale'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
# small_texts = df_labled['lemmatized_tokens_finale'].values
# small_labels = df_labled['score'].values

# # Création d'un modèle - pipeline TF-IDF suivi d'un Naive Bayes
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# # Entraînement du modèle avec les données étiquetées
# model.fit(small_texts, small_labels)

# # Prédire sur la grande dataset
# unlabeled_texts = [' '.join(tokens) for tokens in df['lemmatized_tokens_finale'].values]

# # Utiliser le modèle pour prédire les étiquettes de la grande dataset
# large_predicted_labels = model.predict(unlabeled_texts)

# # Ajouter les étiquettes prédites à la grande dataset
# df = df.copy()
# df['predicted_score_TfidfVectorizer'] = large_predicted_labels


# # Semi-supervised Learning with XGBoost
# 

# In[115]:


# import numpy as np
# import xgboost as xgb
# from gensim.models import Word2Vec
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # Créer des embeddings de mots avec Word2Vec
# sentences = [text.split() for text in df_labled['lemmatized_tokens_finale'].values]
# word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
# word2vec_model.save("xgbmodel/word2vec.model")

# # Convertir les textes en représentations vectorielles en utilisant les embeddings de mots
# def text_to_vector(text, model):
#     words = text.split()
#     vector = np.zeros(model.vector_size)
#     for word in words:
#         if word in model.wv:
#             vector += model.wv[word]
#     return vector / (len(words) + 1e-5)

# X = np.array([text_to_vector(text, word2vec_model) for text in df_labled['lemmatized_tokens_finale'].values])
# small_labels = df_labled['score'].values

# # Encodez les étiquettes en valeurs numériques
# label_encoder = preprocessing.LabelEncoder()
# encoded_labels = label_encoder.fit_transform(small_labels)

# # Diviser les données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# # Entraîner un modèle XGBoost
# params = {
#     'objective': 'multi:softmax',
#     'num_class': len(label_encoder.classes_),
#     'max_depth': 8,  # Augmentation de la profondeur
#     'eta': 0.1,  # Augmentation du taux d'apprentissage
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,  # Ajout d'un hyperparamètre
#     'min_child_weight': 1  # Ajout d'un hyperparamètre
# }
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
# bst = xgb.train(params, dtrain, num_boost_round=200, early_stopping_rounds=10, evals=[(dtest, 'eval')])

# # Évaluer les performances sur l'ensemble de test
# y_pred = bst.predict(dtest)
# print(classification_report(y_test, y_pred))

# # Préparer les données non étiquetées
# X_unlabeled = np.array([text_to_vector(' '.join(tokens), word2vec_model) for tokens in df['lemmatized_tokens_finale'].values])

# # Utiliser le modèle pour prédire les étiquettes de la grande dataset
# d_unlabeled = xgb.DMatrix(X_unlabeled)
# large_predictions = bst.predict(d_unlabeled)

# # Convertir les prédictions en étiquettes lisibles
# large_predicted_labels = label_encoder.inverse_transform(large_predictions.astype(int))

# # Create a copy of the DataFrame to avoid SettingWithCopyWarning
# df = df.copy()

# # Add the predicted labels to the DataFrame
# df['predicted_score_XGB'] = large_predicted_labels


# In[116]:


# from sklearn.tree import DecisionTreeClassifier

# # Étape 1: Entraînez un modèle sur les données étiquetées
# small_texts = df_labled['lemmatized_tokens_finale'].values
# small_labels = df_labled['score'].values

# # Encodage des étiquettes
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(small_labels)

# # Construction du modèle d'arbre de décision
# tree_model = DecisionTreeClassifier()

# # Entraînement du modèle
# tree_model.fit(small_X, encoded_labels)

# # Étape 2: Utilisez le modèle pour prédire des étiquettes pour les données non étiquetées
# unlabeled_texts = [' '.join(tokens) for tokens in df['lemmatized_tokens_finale'].values]
# unlabeled_sequences = tokenizer.texts_to_sequences(unlabeled_texts)
# unlabeled_X = pad_sequences(unlabeled_sequences, maxlen=small_X.shape[1])

# # Prédiction
# predicted_labels = tree_model.predict(unlabeled_X)
# predicted_labels = label_encoder.inverse_transform(predicted_labels)

# # Étape 3: Combinez les données étiquetées originales avec les données auto-étiquetées
# combined_texts = np.concatenate((small_texts, unlabeled_texts))
# combined_labels = np.concatenate((small_labels, predicted_labels))

# # Encodage des étiquettes combinées
# encoded_combined_labels = label_encoder.transform(combined_labels)

# # Étape 4: Ré-entraînez le modèle sur le nouvel ensemble de données combiné
# tree_model.fit(combined_X, encoded_combined_labels)

# # Prédiction sur la grande dataset
# large_unlabeled_texts = [' '.join(tokens) for tokens in df['lemmatized_tokens_finale'].values]
# large_unlabeled_sequences = tokenizer.texts_to_sequences(large_unlabeled_texts)
# large_unlabeled_X = pad_sequences(large_unlabeled_sequences, maxlen=small_X.shape[1])

# # Utilisation du modèle pour prédire les étiquettes de la grande dataset
# large_predicted_labels = tree_model.predict(large_unlabeled_X)
# large_predicted_labels = label_encoder.inverse_transform(large_predicted_labels)

# # Ajouter les étiquettes prédites à la grande dataset
# df['predicted_score_decision_tree'] = large_predicted_labels


# In[117]:


# from sklearn.ensemble import RandomForestClassifier

# # Construction du modèle de forêt aléatoire
# rf_model = RandomForestClassifier()

# # Entraînement du modèle
# rf_model.fit(small_X, encoded_labels)

# # Prédiction sur les données non étiquetées
# predicted_labels = rf_model.predict(unlabeled_X)
# predicted_labels = label_encoder.inverse_transform(predicted_labels)

# # Ré-entraînement du modèle sur l'ensemble de données combiné
# rf_model.fit(combined_X, encoded_combined_labels)

# # Prédiction sur la grande dataset
# large_predicted_labels = rf_model.predict(large_unlabeled_X)
# large_predicted_labels = label_encoder.inverse_transform(large_predicted_labels)

# # Ajouter les étiquettes prédites à la grande dataset
# df['predicted_score_random_forest'] = large_predicted_labels



# Supposons que 'df' est votre DataFrame d'origine.
# Extraction des colonnes nécessaires
extracted_df = df[['comment', 'predicted_score_lstm']].copy()

# Ajout de la nouvelle colonne 'manual_classification' (ici remplie avec des valeurs nulles pour l'instant)
extracted_df['manual_classification'] = None

# Lire le fichier Excel existant
file_path = '../resources/common/data/Comments_Classification.xlsx'
existing_df = pd.read_excel(file_path)

# Concaténation de l'ancien et du nouveau DataFrame
concat_df = pd.concat([existing_df, extracted_df], ignore_index=True)

# Exporter le DataFrame concaténé vers le fichier Excel
concat_df.to_excel(file_path, index=False)


# lire les deux fichiers Excel dans des DataFrame pandas
df1 = pd.read_excel("../resources/common/data/Comments_Classification.xlsx")
df2 = pd.read_excel("../resources/common/data/labled_comments.xlsx")

# renommer la colonne `score` de df2 en `manual_classification`
df2.rename(columns={'score': 'manual_classification'}, inplace=True)

# ajouter une colonne 'predicted_score_lstm' vide à df2 pour correspondre aux colonnes de df1
df2['predicted_score_lstm'] = None

# aligner les colonnes de df2 avec celles de df1
df2 = df2[df1.columns]

# fusionner les deux DataFrame
merged_df = pd.concat([df1, df2], ignore_index=True)

# écrire le DataFrame fusionné dans un nouveau fichier Excel
merged_df.to_excel("../resources/dev_labo/data/processed/merged_comments.xlsx", index=False)
