# Solemn Simulacrum
A Machine learning algorithm that takes data exported from Facebook messenger conversations and uses it to build a model that can respond to messages how you would. 

This project was made to satisfy, in part, the requirements of "COMP 432: Machine Learning" taught by Professor A. Delong. 

## Theoretical
The algorithm works by training a classifier, and a generator, and pitting them against each other.
This is in essence GAN. The Classifier will be a Naive Bayes

## Installation

## Usage

## License

## Credits

## References
[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
to get a word bag count of words used. 

[Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) to classify text strings.

[Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html) to save the classifier's progress