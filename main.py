import math
from collections import Counter, defaultdict

def tokenize(text):
    return text.lower().replace('.', '').split()

def calculate_probabilities(docs, query, lambda_param=0.5):
    tokenized_docs = [tokenize(doc) for doc in docs]
    tokenized_query = tokenize(query)

    corpus = [word for doc in tokenized_docs for word in doc]
    corpus_size = len(corpus)
    corpus_freq = Counter(corpus)

    results = []

    for idx, doc in enumerate(tokenized_docs):
        doc_size = len(doc)
        doc_freq = Counter(doc)

        log_prob = 0
        for word in tokenized_query:
            p_w_d = (doc_freq[word] / doc_size) if doc_size > 0 else 0
            p_w_c = corpus_freq[word] / corpus_size
            p_w = lambda_param * p_w_d + (1 - lambda_param) * p_w_c

            if p_w > 0:
                log_prob += math.log(p_w)

        results.append((idx, log_prob))

    results.sort(key=lambda x: (-x[1], x[0]))

    return [idx for idx, _ in results]

n = int(input())
documents = [input().strip() for _ in range(n)]
query = input().strip()

sorted_indices = calculate_probabilities(documents, query)

print(sorted_indices)
