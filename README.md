# Comments Classification NLP

This repository contains a 6-week internship NLP project focused on classifying comments from medical representatives. The main objective is to categorize the comments into three classes: 'Client' if the mentioned doctor or pharmacist is a client and prescribes the medicine, 'Test' if the concerned person is still testing it, and 'Réclamation' if there are any claims or complaints.

## Project Structure

The project consists of two notebooks located in the `src` directory:
- `Preprocessing_and_Modeling`: This notebook covers data cleaning tasks such as removing stop words and insignificant words, tokenization, lemmatization, and other preprocessing configurations. We have also implemented several models to classify the comments using a semi-supervised approach like XGBoost, LSTM, and Naive Bayes.
- `Data Visualization`: In this notebook, we identified the least correlated words with our targets and removed them.

The datasets used in the project are sourced from two files:
- `new_comment`: This file contains the data for training and testing our models and includes a single feature, 'comment'.
- `labled_data`: This file contains the pre-labled data with predicted scores. It includes the 'comment' feature and the corresponding 'score' label.

The cleaned data is stored in `resources\dev_labo\data\processed\final_cleaned_Comments`. The final predictions (classification results) can be found in `resources\dev_labo\data\processed\Comments_Classification`.

To clean data from doctor names and medicine names, run `process_new_comments.py`.
Next, run `Data Visualization.ipynb` to obtain the output file `600_mots_moins_freq.csv`, which contains the least frequent words.
Finally, preprocess and train the data by running `Preprocessing_and_Modeling.ipynb`.

To correct predictions, add the correct prediction in the `manual_classification` column in `Comments_Classification.xlsx`, and then run `Preprocessing_and_Modeling.ipynb` to correct the model predictions.

## Project Structure
- Project
  - README.md
  - requirements.txt
  - resources
    - common
      - data
        - 600_mots_moins_freq.csv
        - labled_comments.xlsx
        - tun-names.xlsx
        - all_raw_comments_cleaning.xlsx
        - products.xlsx
    - dev_labo
      - data
        - new
          - comment.xlsx
        - processed
          - Comments_Classification.xlsx
          - cleaned_labled_data.xlsx
          - cleaned_data.xlsx
          - final_cleaned_Comments.xlsx
  - src
    - Data Visualization.ipynb
    - Preprocessing_and_Modeling.ipynb
    - process_new_comments.py
  - src/xgbmodel
    - word2vec.model

## Credits

Internship Members:
- Karim Aloulou
- Nadia Bedhiafi

Internship Supervisor:
- Nizar Ellouze
