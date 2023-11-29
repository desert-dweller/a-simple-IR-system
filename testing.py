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


def put_query(q, positional_index, display=1):
    positions_list = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:
        print(term)
        if term in positional_index.keys():
            print(positional_index[term][1].keys())
            [print(k, v) for k, v in enumerate(positional_index[term][1])]
            for doc_id, positions in enumerate(positional_index[term][1]):
                print(positions)
                if positions_list[positions]:
                    print(positions_list)
                    if positions_list[positions][-1] == positions - 1:
                        # print(positions_list[key])
                        positions_list[positions].append(positions)
                else:
                    positions_list[positions].append(positions)
                print(positions_list)
    positions = []
    if display == 1:
        for pos, list in enumerate(positions_list, start=1):
            if len(list) == len(q):
                positions.append('document ' + str(pos))
        return positions
    else:
        for pos, list in enumerate(positions_list, start=1):
            if len(list) == len(q):
                positions.append('doc' + str(pos))
    return positions


def main():
    # Print the positional index
    positional_index = build_positional_index('docs')
    print(positional_index)

    print(put_query("computer information", positional_index))


if __name__ == '__main__':
    main()
