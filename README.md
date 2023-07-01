# Comments Classification NLP

This repository contains a 6-week internship NLP project focused on classifying comments from medical representatives. The main objective is to categorize the comments into three classes: 'Client' if the doctor or pharmacist mentioned is a client and prescribes the medicine, 'Test' if the concerned person is still testing it, and 'Réclamation' if there are any claims or complaints.

## Project Structure

The project consists of three notebooks located in the `src` directory:
- `Data Preprocessing`: This notebook covers data cleaning tasks such as removing stop words and insignificant words, tokenization, lemmatization, and other preprocessing configurations.
- `Data Modeling`: In this notebook, we have implemented several models to classify the comments using a semi-supervised approach.
- `Data Visualization`: In this notebook, we discovred the least corrolated words with our targets, so that we removed them.


The datasets used in the project are sourced from two files:
- `new_comment`: This file contains the data for training and testing our models and includes a single feature, 'comment'.
- `labled_data`: This file contains the pre-labeled data with predicted scores. It includes the 'comment' feature and the corresponding 'score' label.

The cleaned data is stored in `ressources/common/CleanedData/cleaned_comments`, while the lemmatized comments are saved in `ressources/common/CleanedData/Lematized_Comments`. The final predictions (classification results) can be found in `ressources/common/data/comments_classification`.

## Credits
 
Internship Members: 
- Karim Aloulou
- Nadia Bedhiafi

Internship Supervisor:
- Nizar Ellouze
