Sentiment Analysis for Indonesian Language, developed and maintain by Research Centre of Informatics.
Lexicon Approach: 
>>> from sentiID import DictSentiment
>>> mysenti =DictSentiment()
>>> mysenti.debug("Sentence to predict")
>>> mysenti.predict("Sentence to predict")

Machine Learning Approach
Features: TF-IDF, Term Presence, Lexicon Valence
Methods: Ensemble method (Voting Approach) of SVM, Multinomial Naive Bayes, Logistic Regression, and Decision Tree) 
>>> from sentiID import MLSentiment
>>> mysenti =MLSentiment()
>>> mysenti.predict("Sentence to predict")

## Update 1.0.3
- Category : Added More Category

## Update 1.0.2
- Category : Telco (Update), fnb, mie, snack

## Update 1.0.1
- Added Category for Model
- Category : Telco, Generic