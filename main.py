import os

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# PREPROCESSING
def preprocessing(lines):
    tokens = word_tokenize(lines)
    tokens = [t.lower() for t in tokens]  # add the lower-case tokens to the set

    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
    return stemmed_tokens


def read_file(file):
    with open(f'docs/{file}', 'r') as f:
        lines = f.read()
        return lines


def build_positional_index(path):
    # Initialize the positional index
    positional_index = {}

    for doc_id, file in enumerate(os.listdir(path)):
        lines = read_file(file)
        terms = preprocessing(lines)

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
    return positional_index


def put_query(q, positional_index):
    matched_documents = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:
        print(term)
        if term in positional_index.keys():
            for key in positional_index[term][1].keys():
                # print(key)
                # print(positional_index[term][1])
                if matched_documents[key]:
                    print(matched_documents)
                    print(positional_index[term][1][key][0])
                    if matched_documents[key][-1] == positional_index[term][1][key][0] - 1:
                        print(matched_documents[key])
                        matched_documents[key].append(positional_index[term][1][key][0])
                else:
                    matched_documents[key].append(positional_index[term][1][key][0])
                print(matched_documents)

    relevant_docs = []
    for pos, list in enumerate(matched_documents, start=1):
        if len(list) == len(q):
            relevant_docs.append('document ' + str(pos))
    return relevant_docs


def main():
    # Print the positional index
    positional_index = build_positional_index('docs')
    print(positional_index)

    print(put_query("computer information", positional_index))


if __name__ == '__main__':
    main()
