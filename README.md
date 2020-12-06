# Solemn Simulacrum
The goal for Solemn Simulacrum was to take in a person's facebook data, more specifically their messenger conversation history, and create a program that can generate a response to a prompt similar to what that user would would send. We use Keras’ word processing class to convert the message history into a vocabulary of vectors. When processing a sentence, we convert each word into a vector dimension and concatenate them together to make a sentence vector. This is what is used to evaluate a sentence as the simulated user or a random person. We managed to train a GAN, but we have not been able to generate intelligible sentences. 

This project was made to satisfy, in part, the requirements of "COMP 432: Machine Learning" taught by Professor A. Delong. 

## Theoretical


## Installation
```bash
pip install -U python-dotenv
pip install -U numpy
pip install -U nltk
pip install -U tensorflow
pip install -U keras
```
## Usage
Run `main.ipynb`, follow the instructions outlining the code cells

## License

## Credits
[Massimo Triassi](https://github.com/m-triassi)
[Evan Dimopoulos](https://github.com/EvanDime)
## References
// TODO: Update...

[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
to get a word bag count of words used. 
[Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) to classify text strings.
[Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html) to save the classifier's progress
