import os
from collections import Counter

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

unique_terms = set()
document_names = []
word_dictionary = []
query_dictionary = []


# PREPROCESSING
def preprocessing(lines, dictionary):
    tokens = word_tokenize(lines)
    tokens = [t.lower() for t in tokens]  # add the lower-case tokens to the set

    count_docs(tokens, dictionary)
    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
    return stemmed_tokens


def read_file(file):
    with open(f'docs/{file}', 'r') as f:
        lines = f.read()
        document_names.append(file[:-4])
        print(document_names)
        return lines


def build_positional_index(path):
    # Initialize the positional index
    positional_index = {}
    for doc_id, file in enumerate(os.listdir(path)):
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

    print(positional_index)
    normalized_doc_df = tf(word_dictionary, document_names)
    return positional_index, normalized_doc_df


def put_query(q, positional_index):
    matches = [[] for i in range(10)]
    q = preprocessing(q, query_dictionary)
    for term in q:
        print(term)
        if term in positional_index.keys():
            for key in positional_index[term][1].keys():
                # print(key)
                # print(positional_index[term][1])
                if matches[key]:
                    print(matches)
                    print(positional_index[term][1][key][0])
                    if matches[key][-1] == positional_index[term][1][key][0] - 1:
                        print(matches[key])
                        matches[key].append(positional_index[term][1][key][0])
                else:
                    matches[key].append(positional_index[term][1][key][0])
                print(matches)

    matched_docs = []
    for pos, list in enumerate(matches, start=0):
        if len(list) == len(q):
            matched_docs.append(document_names[pos])

    query_df = tf(query_dictionary)
    return matched_docs, query_df


def count_docs(terms, dictionary):
    unique_terms.update(terms)
    document_dictionary = dict(Counter(terms))
    dictionary.append(document_dictionary)
    # for word in terms:
    #     word_dictionary[word] += 1
    print(word_dictionary)


def tf(terms_dictionary, index=["tf-raw"]):
    print(terms_dictionary, index)
    term_frequency_df = pd.DataFrame(terms_dictionary, index=index)
    term_frequency_df.fillna(0, inplace=True)
    term_frequency_df = term_frequency_df.transpose()
    print(term_frequency_df)
    #
    weighted_tf_df = term_frequency_df.applymap(weighted)
    print(weighted_tf_df)

    doc_frequency = term_frequency_df.sum(axis=1)
    idf = len(document_names) / doc_frequency

    inverse_doc_freq = pd.concat([doc_frequency, idf], axis=1)
    inverse_doc_freq.columns = ['df', 'idf']
    print(inverse_doc_freq)

    tf_idf = term_frequency_df.multiply(idf, axis=0)
    print(tf_idf)

    doc_length = np.sqrt((tf_idf ** 2).sum())
    print(doc_length)

    normalized_tf_idf = tf_idf / doc_length
    print(normalized_tf_idf)

    if index[0] == 'tf-raw':
        query_df = pd.concat([term_frequency_df, weighted_tf_df, idf, term_frequency_df, normalized_tf_idf], axis=1)
        query_df.columns = ['tf-raw', 'weighted-tf', 'idf', 'tf-idf', 'normalized']
        print("query length:", doc_length)
        print(query_df)
        return query_df

    return normalized_tf_idf


def similarity(query_df, normalized_tf_idf, matched_docs):
    query_terms = query_df.index
    matched_docs_df = normalized_tf_idf.loc[query_terms, matched_docs]
    print(matched_docs_df)

    query_normalized = query_df.loc[:, 'normalized']
    product_df = matched_docs_df.multiply(query_normalized, axis=0)
    print(product_df)

    similarity_score = product_df.sum().sort_values(ascending=False)
    print(similarity_score)


def weighted(x):
    if x > 0:
        return np.log10(x) + 1
    return 0


def main():
    # Print the positional index
    positional_index, normalized_doc_df = build_positional_index('docs')

    q = "computer information"
    matched_docs, query_df = put_query(q, positional_index)

    similarity(query_df, normalized_doc_df, matched_docs)


if __name__ == '__main__':
    main()
