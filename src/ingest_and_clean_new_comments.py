#!/usr/bin/env python
# coding: utf-8

#Installer tous les packages nécaissaires
# pip install -r ../requirements.txt
#ou bien
# !pip install autocorrect
# !pip install -U spacy
# !python -m spacy download fr_core_news_sm
# !pip install pyspellchecker
# !pip install keras==2.11.0
# !pip install xgboost
import shutil
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



# Chemin du dossier source
source_folder = '../resources/dev_labo/data/new/All Comments'

# Chemin du fichier de destination
dest_file = '../resources/dev_labo/data/new/comment.xlsx'

# Initialiser une liste pour stocker toutes les données
all_data = []

# Parcourir tous les fichiers dans le dossier source
for filename in os.listdir(source_folder):
    # Si le fichier est un fichier Excel
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # Lire les données dans le fichier
        df = pd.read_excel(os.path.join(source_folder, filename))
        
        # Prendre la première colonne et convertir en DataFrame
        df_single_column = pd.DataFrame(df.iloc[:, 0])
        
        # Renommer la colonne pour assurer la compatibilité lors de la concaténation
        df_single_column.columns = ['comment']
        
        # Ajouter le dataframe à la liste
        all_data.append(df_single_column)

# Concaténer tous les dataframes
merged_data = pd.concat(all_data, ignore_index=True)

# Écrire toutes les données dans le fichier de destination
merged_data.to_excel(dest_file, index=False)



# Chemins des fichiers XLSX
file1 = '../resources/dev_labo/data/new/comment.xlsx'
file2 = '../resources/common/data/products.xlsx'
file3 = '../resources/common/data/tun-names.xlsx'

# Lecture des fichiers XLSX
dfcomment = pd.read_excel(file1)
dfproducts = pd.read_excel(file2)

# Lecture de la première feuille du fichier 3
wb = openpyxl.load_workbook(file3)
sheet_names = wb.sheetnames
dffirst_name = pd.read_excel(file3, sheet_name=sheet_names[0])

# Lecture de la deuxième feuille du fichier 3
dflast_name = pd.read_excel(file3, sheet_name=sheet_names[1])


# In[2]:


#afficher la data frame contenant les prénoms
dffirst_name
#afficher la data frame contenant les noms
dflast_name
#afficher la data frame contenant les commentaires
dfcomment


# In[8]:


#afficher la data frame contenant les noms des produits
dfproducts


# In[3]:


def comment_to_list(comment):
    return comment.split()


# In[4]:


def replace_words(data, df):
    dffirst_name['first_name'] = dffirst_name['first_name'].str.replace(' ', '').str.lower()
    dflast_name['last_name'] = dflast_name['last_name'].str.replace(' ', '').str.lower()
    dfproducts['products'] = dfproducts['products'].str.lower()

    for i, word in enumerate(data):
        if word.lower() in dffirst_name['first_name'].tolist() or word.lower() in dflast_name['last_name'].tolist():
            data[i] = 'prospect'

        match = re.search(r'\b{}\b'.format(re.escape(word.lower())), ' '.join(dfproducts['products'].tolist()))
        if match:
            data[i] = 'produit'

        # Supprimer les mots spécifiques
        if word.lower() == "c'est" or word.lower() == 'plus':
            data[i] = ''

    # Réduire les occurrences consécutives de "produit" à une seule occurrence
    reduced_data = []
    prev_word = None
    for word in data:
        if word == 'produit' and prev_word == 'produit':
            continue  # Ignorer les mots consécutifs "produit"
        if word != '':
            reduced_data.append(word)
        prev_word = word

    return reduced_data

# In[5]:


def process_comments(df):
    for i in range(len(dfcomment['comment'])):
        if isinstance(dfcomment['comment'][i], str):  # vérifie si le commentaire est une chaîne
            comment_list = comment_to_list(dfcomment['comment'][i])
            new_comment = replace_words(comment_list, df)
            dfcomment.at[i, 'comment'] = ' '.join(new_comment) # assign new_comment as a list
    return df


# In[6]:


dfnew_comment = dfcomment.copy()
dfnew_comment = process_comments(dfcomment)

# In[7]:


# Écrir le DataFrame dans un fichier Excel
dfnew_comment.to_excel('../resources/common/data/all_raw_comments_cleaning.xlsx', index=False)





# # Preprocessing Phase



# Chemins des fichiers XLSX
file1 = '../resources/common/data/all_raw_comments_cleaning.xlsx'
file4 = '../resources/common/data/labled_comments.xlsx'

# Lecture des fichiers XLSX
dfnew_comment = pd.read_excel(file1)
df_labled = pd.read_excel(file4)
def clean_text(text):
    # Vérifiez si le texte est une chaîne
    if isinstance(text, str):
        # Supprimer le texte de retweet de style ancien "RT"
        text = re.sub(r'^RT[\s]+', '', text)
        # Supprimer les hyperliens
        text = re.sub(r'https?://[^\s\n\r]+', '', text)
        # Supprimer les hashtags (seulement supprimer le signe de hash # du mot)
        text = re.sub(r'#', '', text)
        # Supprimer les dates au format AAAA-MM-JJ
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
        # Supprimer l'heure au format HH:MM ou HH:MM:SS
        text = re.sub(r'\b\d{2}:\d{2}(:\d{2})?\b', '', text)
        # Supprimer les caractères spéciaux
        text = re.sub(r'[^\w\s]', '', text)
        # Supprimer les lignes vides ou les lignes avec juste un point
        text = re.sub(r'^(\s*\.?\s*)$', '', text, flags=re.MULTILINE)
        # Supprimer les accents
        text = unidecode(text)
    else:
        # Si le texte est un nombre, convertissez-le en chaîne de caractères
        if isinstance(text, (int, float)):
            text = str(text)
        # Si le texte est une valeur NaN, remplacez-le par une chaîne vide
        elif pd.isnull(text):
            text = ''
    return text.strip()  # Supprimer les espaces blancs en tête ou en queue
# Appliquer la fonction à chaque élément du DataFrame
dfnew_comment = dfnew_comment.applymap(clean_text)
# Remplacer les lignes qui sont juste un point ou une virgule (maintenant une chaîne vide après avoir supprimé les caractères spéciaux) par NaN
dfnew_comment.replace("", np.nan, inplace=True)
# Supprimer les lignes avec des valeurs NaN
dfnew_comment.dropna(subset=['comment'], inplace=True)
df_labled['comment'] = df_labled['comment'].apply(clean_text)
# Remplacer les lignes qui sont juste un point ou une virgule (maintenant une chaîne vide après avoir supprimé les caractères spéciaux) par NaN
df_labled.replace("", np.nan, inplace=True)
# Supprimer les lignes avec des valeurs NaN dans la colonne 'comment'
df_labled.dropna(subset=['comment'], inplace=True)
# Renommer dfnew_comment par df
df=dfnew_comment
# # Tokenization 
# instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
# Tokeniser les textes dans la colonne 'comment'
df['tokens'] = df['comment'].apply(tokenizer.tokenize)
df_labled['tokens'] = df_labled['comment'].apply(tokenizer.tokenize)
# Afficher le DataFrame après tokenisation
nltk.download('stopwords')
stopwords_french = stopwords.words('french')
df_clean1 = []
# Parcourir chaque liste de tokens
for tokens in df['tokens']:
    clean_tokens = []  # Liste pour stocker les tokens nettoyés d'un texte particulier
    for word in tokens:  # Parcourir chaque mot dans la liste de tokens
        # Vérifier si le mot n'est pas un mot d'arrêt et n'est pas un signe de ponctuation
        if word == 'pas' or word == 'ne' or (word not in stopwords_french and word not in string.punctuation):
            clean_tokens.append(word)
    # Ajouter les tokens nettoyés de ce texte à la liste df_clean
    df_clean1.append(clean_tokens)

# Vous pouvez maintenant ajouter df_clean comme une nouvelle colonne à votre DataFrame
df['clean_tokens'] = df_clean1
df_clean2 = []
for tokens in df_labled['tokens']:
    clean_tokens = []
    for word in tokens:
        if word == 'pas' or word == 'ne' or (word not in stopwords_french and word not in string.punctuation):
            clean_tokens.append(word)
    df_clean2.append(clean_tokens)

df_labled['clean_tokens'] = df_clean2
# Créer une nouvelle colonne 'clean_data_noTokenized' en rejoignant les tokens nettoyés en une seule chaîne de caractères
df['clean_data_noTokenized'] = df['clean_tokens'].apply(' '.join)
df_labled['clean_data_noTokenized'] = df_labled['clean_tokens'].apply(' '.join)
# # Lemmatizing
# Charger le modèle de langue français de spaCy
nlp = spacy.load('fr_core_news_sm')
# Créer une liste vide pour stocker les tokens lemmatisés
df_lemmatized1 = []
# Parcourir chaque liste de tokens
for tokens in df_clean1:
    lemmatized_tokens = []  # Liste pour stocker les tokens lemmatisés d'un texte particulier
    for word in tokens:  # Parcourir chaque mot dans la liste de tokens
        # Lemmatisation du mot
        doc = nlp(word)
        lemma = doc[0].lemma_ if doc else word
        lemmatized_tokens.append(lemma)  # Ajouter à la liste
    # Ajouter les tokens lemmatisés de ce texte à la liste df_lemmatized
    df_lemmatized1.append(lemmatized_tokens)
    
    
for i, text in enumerate(df_lemmatized1):
    text_str = ' '.join(text)
    text_str = re.sub(r'\bprescr\w*\b', 'prescrire', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bsat\b|\bst\b|\bsatisfaire\b|\bsatisfaisant\b|\bsatisfaites\b', 'satisfait', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\brap(el|elle|ell)\b', 'rappel', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bbc\b|\bbcp\b', 'beaucoup', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\baimer\b|\baime\b', 'aimer', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\btre\b', 'tres', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bretou\b', 'retour', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blindication\b', 'indication', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bpresentationell\b', 'presentation', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blutilis\b', 'utiliser', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blachete\b', 'acheter', text_str, flags=re.IGNORECASE)

    # Remettre la chaîne de texte modifiée dans la liste df_lemmatized
    df_lemmatized1[i] = text_str.split()

# Vous pouvez maintenant ajouter df_lemmatized comme une nouvelle colonne à votre DataFrame
df['lemmatized_tokens'] = df_lemmatized1
# Afficher le DataFrame
nlp = spacy.load('fr_core_news_sm')
# Créer une liste vide pour stocker les tokens lemmatisés
df_lemmatized2 = []
# Parcourir chaque liste de tokens
for tokens in df_clean2:
    lemmatized_tokens = []  # Liste pour stocker les tokens lemmatisés d'un texte particulier
    for word in tokens:  # Parcourir chaque mot dans la liste de tokens
        # Lemmatisation du mot
        doc = nlp(word)
        lemma = doc[0].lemma_ if doc else word
        lemmatized_tokens.append(lemma)  # Ajouter à la liste
    # Ajouter les tokens lemmatisés de ce texte à la liste df_lemmatized
    df_lemmatized2.append(lemmatized_tokens)
    
    
for i, text in enumerate(df_lemmatized2):
    # Convertir la liste de tokens en une seule chaîne de texte
    text_str = ' '.join(text)
    text_str = re.sub(r'\bprescr\w*\b', 'prescrire', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bsat\b|\bst\b|\bsatisfaire\b|\bsatisfaisant\b|\bsatisfaites\b', 'satisfait', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\brap(el|elle|ell)\b', 'rappel', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bbc\b|\bbcp\b', 'beaucoup', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\baimer\b|\baime\b', 'aimer', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\btre\b', 'tres', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bretou\b', 'retour', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blindication\b', 'indication', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\bpresentationell\b', 'presentation', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blutilis\b', 'utiliser', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\blachete\b', 'acheter', text_str, flags=re.IGNORECASE)

    # Remettre la chaîne de texte modifiée dans la liste df_lemmatized
    df_lemmatized2[i] = text_str.split()

# Vous pouvez maintenant ajouter df_lemmatized comme une nouvelle colonne à votre DataFrame
df_labled['lemmatized_tokens'] = df_lemmatized2
# Lire le fichier mots.csv et créer un ensemble de mots à supprimer
words_to_remove = set()
with open('../resources/common/data/600_mots_moins_freq.csv', 'r') as file:
    for line in file:
        word = line.strip()  # supprime les espaces en début et en fin de ligne
        words_to_remove.add(word)
# Pour chaque ligne de df['lemmatized_tokens'], supprimez les mots présents dans mots.csv
def filter_tokens(tokens):
    return [token for token in tokens if token not in words_to_remove]
# Appliquer la fonction de filtrage à chaque ligne de lemmatized_tokens
df['lemmatized_tokens_finale'] = df['lemmatized_tokens'].apply(filter_tokens)
df_labled['lemmatized_tokens_finale'] = df_labled['lemmatized_tokens'].apply(filter_tokens)
# fonction pour juger la qualité
def judge_quality(comment):
    if len(comment) < 5:
        return 'bad'
    else:
        return 'good'
# Création de la nouvelle DataFrame avec seulement les colonnes nécessaires
new_df = df[['comment', 'lemmatized_tokens_finale']].copy()
# Création des nouvelles colonnes 'quality' et 'manual_clean_comments' (ici remplie avec des valeurs nulles pour l'instant)
# Appliquer la fonction à la colonne des commentaires
new_df['quality'] = new_df['comment'].apply(judge_quality)
new_df['manual_clean_comments'] = ''
# Réorganisation des colonnes dans l'ordre que vous voulez
new_df = new_df[['comment', 'quality', 'lemmatized_tokens_finale', 'manual_clean_comments']]
# Enregistrement de la nouvelle DataFrame dans un fichier Excel
new_df.to_excel('../resources/common/data/final_cleaned_Comments.xlsx',index=False)
new_df.to_pickle('../resources/common/data/final_cleaned_Comments.pkl')# Charger les deux fichiers Excel dans des DataFrames
comments_classification_df = pd.read_excel('../resources/common/data/Comments_Classification.xlsx')
# Trouver les indices des lignes où 'manual_classification' est non nulle
indices = comments_classification_df[comments_classification_df['manual_classification'].notna()].index

# Pour chaque indice, copier la ligne correspondante dans labeled_df et la supprimer de comments_classification_df
for i in indices:
    row = comments_classification_df.loc[i]
    new_row = pd.DataFrame({'comment': [row['comment']], 'score': [row['manual_classification']]})
    df_labled = pd.concat([df_labled, new_row], ignore_index=True)
    comments_classification_df = comments_classification_df.drop(i)

# Sauvegarder les DataFrames mis à jour dans leurs fichiers Excel respectifs
# df_labled.to_excel('../resources/dev_labo/data/processed/cleaned_labled_data.xlsx', index=False)
comments_classification_df.to_excel('../resources/common/data/Comments_Classification.xlsx', index=False)
df_labled.to_pickle('../resources/common/data/final_cleaned_labled_Comments.pkl')

source_dir = '../resources/dev_labo/data/new/All Comments'
destination_dir = '../resources/dev_labo/data/processed'
# Parcours de tous les fichiers et dossiers dans le répertoire source
for root, dirs, files in os.walk(source_dir):
    # Parcours de tous les fichiers
    for file in files:
        # Chemin complet du fichier source
        source_file = os.path.join(root, file)
        # Chemin complet du fichier de destination
        destination_file = os.path.join(destination_dir, file)
        # Déplacement du fichier source vers le répertoire de destination
        shutil.move(source_file, destination_file)

print("Déplacement terminé.")






