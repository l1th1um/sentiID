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
