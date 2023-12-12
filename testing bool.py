import os
from collections import Counter

import numpy as np
import pandas as pd
from natsort import natsorted
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

unique_terms = set()
document_names = []
word_dictionary = []
query_dictionary = []
document_idf = pd.DataFrame()


# PREPROCESSING
def preprocessing(lines, dictionary):
    tokens = word_tokenize(lines)
    tokens = [t.lower() for t in tokens]  # add the lower-case tokens to the set

    count_docs(tokens, dictionary)  # count
    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
    return stemmed_tokens


def count_docs(terms, dictionary):
    unique_terms.update(terms)

    # count the term frequency in each document
    document_dictionary = dict(Counter(terms))
    dictionary.append(document_dictionary)


def read_file(file):
    with open(f'docs/{file}', 'r') as f:
        lines = f.read()
        document_names.append('d' + file[:-4])
        return lines


def build_positional_index(path):
    # Initialize the positional index
    positional_index = {}
    for doc_id, file in enumerate(natsorted(os.listdir(path)), start=0):
        lines = read_file(file)
        terms = preprocessing(lines, word_dictionary)

        for i, token in enumerate(terms):
            # create a new posting list if it isn't there yet
            # if token not in positional_index:
            #     positional_index[token] = [0, {}]
            positional_index.setdefault(token, [0, {}])

            # get the existing posting list of the term, ex: {0: [0]} for the term {'comput': {0: [0]}}
            # term_list = positional_index.get(token, {})
            term_list = positional_index[token]

            # get the existing position list if that term appeared in a document, ex: [] if the doc_id is 1
            term_positions = term_list[1].get(doc_id, [])
            term_positions.append(i)

            # increment the document frequency by 1 if the document isn't in the posting list yet
            if doc_id not in positional_index[token][1]:
                positional_index[token][0] += 1

            # append the new positions to the posting list, ex: it becomes 1: [0]
            # positional_index[token][1][doc_id] = term_positions
            term_list[1][doc_id] = term_positions

    print('\n', 'Unique Set of Terms', sep='')
    print("-" * 50)
    print(unique_terms, '\n')

    # making the positional index a dataframe to print it out clearer
    temp = pd.DataFrame(positional_index, index=['document frequency', 'positional posting lists'], )
    print("Positional Index")
    print("-" * 50)
    print(temp.transpose(), '\n')
    normalized_doc_df = tf(word_dictionary, 'Document')
    return positional_index, normalized_doc_df


def phrase_query(q, positional_index, normalized_doc_df):
    matches = [[] for _ in range(10)]
    q = preprocessing(q, query_dictionary)

    for term in q:
        if term in positional_index.keys():
            for doc_id in positional_index[term][1].keys():
                if matches[doc_id]:
                    if matches[doc_id][-1] == positional_index[term][1][doc_id][0] - 1:
                        matches[doc_id].append(positional_index[term][1][doc_id][0])
                else:
                    matches[doc_id].append(positional_index[term][1][doc_id][0])

    matched_docs = []
    for doc_id, match in enumerate(matches):
        if len(match) == len(q):
            matched_docs.append(document_names[doc_id])
    return matched_docs


def split_boolean_query(query):
    operators = ['NOT', 'AND', 'OR']
    phrases = []
    operator_stack = []
    current_phrase = []

    for word in query.split():
        if word not in operators:
            current_phrase.append(word)
        else:
            if current_phrase:
                phrases.append(" ".join(current_phrase))
                current_phrase.clear()
            if word == 'NOT':
                current_phrase.append(word)
            else:
                operator_stack.append(word)
        print(current_phrase)
        print(phrases)
    if current_phrase:
        phrases.append(" ".join(current_phrase))

    return phrases, operator_stack


def put_query(q, positional_index, normalized_doc_df):
    # split the boolean query into phrases and operators
    phrases, operators = split_boolean_query(q)
    print(phrases, operators)
    # Initialize an empty set to store matched documents
    matched_docs = set()

    # perform phrase queries and handle NOT operators
    for i, phrase in enumerate(phrases):
        is_not_term = False
        if phrase.startswith('NOT'):
            is_not_term = True
            phrase = phrase[4:]

        phrase_matches = set(phrase_query(phrase, positional_index, normalized_doc_df))
        # Apply NOT operator if needed
        if is_not_term:
            phrase_matches = set(document_names) - phrase_matches
        print(phrase, phrase_matches)
        if operators and i != 0:  # perform operators after first phrase
            operator = operators.pop(0)
            print(operator)
            if operator == 'AND':
                matched_docs &= phrase_matches
            elif operator == 'OR':
                matched_docs |= phrase_matches
        else:
            matched_docs = set(phrase_matches)
        print(matched_docs)

    if not q:  # if query is empty
        print("Seems like you entered an empty query")
    elif not matched_docs:  # or no matched documents
        print("Results: No matched documents")
        print("Try again with different terms\n")
    else:
        matched_docs = natsorted(list(matched_docs))
        query_df = tf(query_dictionary, 'Query')
        similarity(query_df, normalized_doc_df, matched_docs)

    return matched_docs


def tf(terms_dictionary, document_type):
    global document_idf
    global query_dictionary
    if document_type == 'Document':
        index = document_names
        N = len(document_names)
    else:
        index = ['tf-raw']
        N = len(terms_dictionary[0])
        print(terms_dictionary)
        t = {}
        for query_phrase in terms_dictionary:
            t.update(query_phrase)
        terms_dictionary = [t]

    print('\n', "*" * 100, sep='')
    print("\t" * 10, document_type)
    print("*" * 100, '\n')

    print(len(terms_dictionary))
    term_frequency_df = pd.DataFrame(terms_dictionary, index=index)
    # term_frequency_df = pd.DataFrame(terms_dictionary, columns=positional_index.index, index=index)
    term_frequency_df.fillna(0, inplace=True)

    term_frequency_df = term_frequency_df.transpose()
    original_index = term_frequency_df.index
    # l = []
    # for token in original_index:
    #     l.append(PorterStemmer().stem(token))
    term_frequency_df.index = [PorterStemmer().stem(token) for token in original_index]
    print(term_frequency_df)
    weighted_tf_df = term_frequency_df.applymap(weighted)

    doc_frequency = term_frequency_df.sum(axis=1)
    if document_type == 'Document':
        idf = np.log10(N / doc_frequency)
    else:
        # removing all the terms that don't occur in documents, otherwise it ruins the dictionaries
        term_index = [term for term in term_frequency_df.index if term in document_idf.index]
        print(term_index)

        # get the idf values of the terms in the query
        print(term_frequency_df.index)
        print(document_idf)
        idf = document_idf.loc[term_index, 'idf']
    print(idf)
    inverse_doc_freq = pd.concat([doc_frequency, idf], axis=1)
    inverse_doc_freq.columns = ['df', 'idf']
    tf_idf = term_frequency_df.multiply(inverse_doc_freq['idf'], axis=0)
    tf_idf.fillna(0.0, inplace=True)
    print("tf-idf", tf_idf)
    doc_length = np.sqrt((tf_idf ** 2).sum())

    normalized_tf_idf = tf_idf / doc_length

    if document_type == 'Query':
        # empty query dictionary incase of another incoming query
        query_dictionary = []
        query_df = pd.concat([term_frequency_df, weighted_tf_df, idf, tf_idf, normalized_tf_idf], axis=1)
        query_df.columns = ['tf-raw', 'weighted-tf', 'idf', 'tf-idf', 'normalized']
        print(f"Normalized {document_type} TF-IDF Matrix")
        print("-" * 50)
        print(query_df)
        print("query length:", doc_length.values, '\n')
        return query_df
    else:
        print("Term Frequency Matrix")
        print("-" * 50)
        print(term_frequency_df, '\n')
        print("Weighted Term Frequency (1+log(tf)) Matrix")
        print("-" * 50)
        print(weighted_tf_df, '\n')
        print(f"Inverse {document_type} Frequency (IDF) Matrix")
        print("-" * 50)
        print(inverse_doc_freq, '\n')
        print("TF-IDF Matrix")
        print("-" * 50)
        print(tf_idf.to_string(), '\n')
        print("Document Lengths")
        print("-" * 50)
        print(doc_length.to_string(), '\n')
        print(f"Normalized {document_type} TF-IDF Matrix")
        print("-" * 50)
        print(normalized_tf_idf.to_string(), '\n')
        document_idf = inverse_doc_freq
        return normalized_tf_idf


def weighted(x):
    if x > 0:
        return np.log10(x) + 1
    return 0


def similarity(query_df, normalized_tf_idf, matched_docs):
    print('\n', "*" * 100, sep='')
    print("\t" * 10, "Post-Query")
    print("*" * 100, '\n')
    # removing all the terms that don't occur in documents, otherwise it ruins the dictionaries

    query_terms = [term for term in query_df.index if term in document_idf.index]
    print(query_terms)
    print(matched_docs)
    print(normalized_tf_idf.loc[:, matched_docs])

    matched_docs_df = normalized_tf_idf.loc[query_terms, matched_docs]
    print("Matched Documents")
    print("-" * 50)
    print(matched_docs_df.columns.values, '\n')

    query_normalized = query_df.loc[:, 'normalized']
    product_df = matched_docs_df.multiply(query_normalized, axis=0)
    product_df.fillna(0.0, inplace=True)
    print("Product (Query * Matched Documents) Matrix")
    print("-" * 50)
    print(product_df, '\n')

    similarity_score = product_df.sum().sort_values(ascending=False)
    print("Similarity Score")
    print("-" * 50)
    print(similarity_score.to_string(), '\n')


def main():  # Print the positional index
    positional_index, normalized_doc_df = build_positional_index('docs')

    while True:
        q = input("Enter your query: ")
        matched_docs = put_query(q, positional_index, normalized_doc_df)

        flag = input("Would you like to end the program? (Q to quit): ")
        if flag.lower() == 'q':
            break


main()
