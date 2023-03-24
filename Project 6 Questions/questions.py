import math
import nltk
import os
import string
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = {}

    for i in os.listdir(directory):
        with open(os.path.join(directory, i), mode="r", encoding="utf-8") as file:
            result[i] = file.read()

    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized_words = nltk.tokenize.word_tokenize(document.lower())

    result = []

    for i in tokenized_words:
        if i not in set(string.punctuation) and i not in set(nltk.corpus.stopwords.words("english")):
            result.append(i)

    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    frequency_dict = {}

    for i in documents:
        for j in set(documents[i]):
            if j in frequency_dict.keys():
                frequency_dict[j] += 1
            else:
                frequency_dict[j] = 1

    result = {}

    for k in frequency_dict:
        result[k] =  math.log(len(documents) / frequency_dict[k])
    
    return result


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf_dict = {}

    for key, value in files.items():
        tfidf = 0
        for i in query:
            if i in value:
                tfidf += value.count(i) * idfs[i]
        if tfidf != 0:
            tfidf_dict[key] = tfidf

    tfidf_dict_sorted = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    
    result = []

    for key, value in tfidf_dict_sorted:
        result.append(key)

    return result[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_dict  = {}

    for key, value in sentences.items():
        idf = 0
        for i in query:
            if i in value:
                idf += idfs[i]
        if idf != 0:
            density = sum([value.count(x) for x in query]) / len(value)
            idf_dict[key] = (idf, density)

    idf_dict_sorted = sorted(idf_dict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    result = []

    for key, value in idf_dict_sorted:
        result.append(key)

    return result[:n]


if __name__ == "__main__":
    main()
