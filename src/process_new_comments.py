#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import openpyxl
import os
import re
import pandas as pd



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
dfnew_comment


# In[7]:


# Écrir le DataFrame dans un fichier Excel
dfnew_comment.to_excel('../resources/common/data/all_raw_comments_cleaning.xlsx', index=False)