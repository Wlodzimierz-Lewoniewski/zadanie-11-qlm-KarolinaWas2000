import math
from collections import Counter, defaultdict

def tokenize(text):
    return text.lower().split()

def calculate_probabilities(documents, query, lambd=0.5):
    tokenized_docs = [tokenize(doc) for doc in documents]

    corpus_word_counts = Counter(word for doc in tokenized_docs for word in doc)
    total_corpus_words = sum(corpus_word_counts.values())

    probabilities = []
    for doc_idx, doc in enumerate(tokenized_docs):
        doc_word_counts = Counter(doc)
        total_doc_words = sum(doc_word_counts.values())

        query_prob = 0
        for term in tokenize(query):
            doc_prob = doc_word_counts[term] / total_doc_words if total_doc_words > 0 else 0
            corpus_prob = corpus_word_counts[term] / total_corpus_words if total_corpus_words > 0 else 0
            smoothed_prob = lambd * doc_prob + (1 - lambd) * corpus_prob
            if smoothed_prob > 0:
                query_prob += math.log(smoothed_prob)
            else:
                query_prob += float('-inf')

        probabilities.append((doc_idx, query_prob))

    return probabilities

def rank_documents(probabilities):
    return [idx for idx, _ in sorted(probabilities, key=lambda x: (-x[1], x[0]))]

def main():
    n = int(input())
    documents = [input for i in range(n)]
    query = input()

    probabilities = calculate_probabilities(documents, query)
    ranked_indices = rank_documents(probabilities)

    print(' '.join(map(str, ranked_indices)))

if __name__ == "__main__":
    main()
