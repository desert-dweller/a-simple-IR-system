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
    document_dictionary = dict(Counter(terms))
    dictionary.append(document_dictionary)
    # for word in terms:
    #     word_dictionary[word] += 1


def read_file(file):
    with open(f'docs/{file}', 'r') as f:
        lines = f.read()
        document_names.append(file[:-4])
        return lines


def build_positional_index(path):
    # Initialize the positional index
    positional_index = {}
    for doc_id, file in enumerate(natsorted(os.listdir(path))):
        lines = read_file(file)
        terms = preprocessing(lines, word_dictionary)

        for i, token in enumerate(terms):
            # create a new posting list if it isn't there yet
            if token not in positional_index:
                positional_index[token] = [0, {}]

            # get the existing posting list of the term, ex: {0: [0]} for the term {'comput': {0: [0]}}
            term_list = positional_index.get(token, {})
            # get the existing position list if that term appeared in a document, ex: [] if the doc_id is 1
            term_positions = term_list[1].get(doc_id, [])

            # append the new position in that list, ex: it becomes 1: [0]
            positional_index[token][0] += 1
            positional_index[token][1][doc_id] = term_positions + [i]

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


def tf(terms_dictionary, document_type):
    global document_idf
    if document_type == 'Document':
        index = document_names
        N = len(document_names)
    else:
        index = ['tf-raw']
        N = len(terms_dictionary[0])

    print('\n', "*" * 100, sep='')
    print("\t" * 10, document_type)
    print("*" * 100, '\n')

    term_frequency_df = pd.DataFrame(terms_dictionary, index=index)
    term_frequency_df.fillna(0, inplace=True)
    term_frequency_df = term_frequency_df.transpose()

    weighted_tf_df = term_frequency_df.applymap(weighted)

    doc_frequency = term_frequency_df.sum(axis=1)
    if document_type == 'Document':
        idf = np.log10(N / doc_frequency)
    else:
        # get the idf values of the terms in the query
        idf = document_idf.loc[term_frequency_df.index, 'idf']

    inverse_doc_freq = pd.concat([doc_frequency, idf], axis=1)
    inverse_doc_freq.columns = ['df', 'idf']
    print(term_frequency_df)
    tf_idf = term_frequency_df.multiply(idf, axis=0)

    doc_length = np.sqrt((tf_idf ** 2).sum())

    normalized_tf_idf = tf_idf / doc_length

    if document_type == 'Query':
        query_df = pd.concat([term_frequency_df, weighted_tf_df, idf, term_frequency_df, normalized_tf_idf], axis=1)
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


def put_query(q, positional_index, normalized_doc_df):
    matches = [[] for i in range(10)]
    q = preprocessing(q, query_dictionary)

    matched_docs = []
    for term in q:
        if term in positional_index.keys():
            for doc_id in positional_index[term][1].keys():
                if matches[doc_id]:
                    if matches[doc_id][-1] == positional_index[term][1][doc_id][0] - 1:
                        matches[doc_id].append(positional_index[term][1][doc_id][0])
                else:
                    matches[doc_id].append(positional_index[term][1][doc_id][0])

    for pos, list in enumerate(matches, start=0):
        if len(list) == len(q):
            matched_docs.append(document_names[pos])

    if not q or not matched_docs:  # if query is empty or no matched documents
        print("Results: No matched documents")
        print("Seems like you entered an invalid query, try again with different terms\n")
    else:
        query_df = tf(query_dictionary, 'Query')
        similarity(query_df, normalized_doc_df, matched_docs)

    return matched_docs


def similarity(query_df, normalized_tf_idf, matched_docs):
    print('\n', "*" * 100, sep='')
    print("\t" * 10, "Post-Query")
    print("*" * 100, '\n')

    query_terms = query_df.index
    matched_docs_df = normalized_tf_idf.loc[query_terms, matched_docs]
    print(matched_docs_df)
    print("Matched Documents")
    print("-" * 50)
    print(matched_docs_df.columns.values, '\n')

    query_normalized = query_df.loc[:, 'normalized']
    product_df = matched_docs_df.multiply(query_normalized, axis=0)
    print("Product (Query * Matched Documents) Matrix")
    print("-" * 50)
    print(product_df, '\n')

    similarity_score = product_df.sum().sort_values(ascending=False)
    print("Similarity Score")
    print("-" * 50)
    print(similarity_score.to_string(), '\n')


def weighted(x):
    if x > 0:
        return np.log10(x) + 1
    return 0


def main():
    # Print the positional index
    positional_index, normalized_doc_df = build_positional_index('docs')

    while True:
        q = input("Enter your query: ")
        matched_docs = put_query(q, positional_index, normalized_doc_df)

        flag = input("Would you like to end the program? (Q to quit): ")
        if flag.lower() == 'q':
            break


main()
