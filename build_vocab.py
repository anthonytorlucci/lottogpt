"""build the vocabulary from input sequences to tokens"""
from pathlib import Path
from collections import deque
import math
import pickle

import numpy
from data.data import lottery_data

def vocabulary_helper3(i_min, i_max, j_min, j_max, k_min, k_max):
    vocab = {}
    cntr = 0
    for i in range(i_min, i_max+1):
        j_start = j_min if i < j_min else i+1
        for j in range(j_start, j_max+1):
            k_start = k_min if j < k_min else j+1
            for k in range(k_start, k_max+1):
                vocab[(i, j, k)] = cntr
                cntr += 1

    vocab_size = len(vocab)
    return vocab, vocab_size

# https://ashukumar27.medium.com/similarity-functions-in-python-aa6dfe721035
def euclidean_distance(p:tuple, q:tuple):
    """Euclidean distance between two points, each point having n dimensions, i.e. len(p) == len(q) == n."""
    return math.sqrt(sum(math.pow(a-b,2) for a, b in zip(p, q)))

def square_rooted(x):
   return math.sqrt(sum([a*a for a in x]))

def cosine_similarity(p:tuple, q:tuple):
    # return numpy.dot(p, q) / (vector_magnitude(p) * vector_magnitude(q))
    numerator = sum(a*b for a,b in zip(p,q))
    denominator = square_rooted(p)*square_rooted(q)
    return numerator/float(denominator)

def similarity_metric(p:tuple, q:tuple):
    return euclidean_distance(p=p,q=q) + (1-cosine_similarity(p=p, q=q))

def vocabulary_tokens_by_similarity(vocab:dict) -> dict:
    seqs = [k for k,v in vocab.items()]
    # print(f"vocab size: {vocab_size}, e.g. sequence length: {len(seqs)}")

    dq = deque([seqs[0]])
    # seqs.pop()  # default is to remove last item, seqs[-1]
    seqs.pop(0)
    # print(dq)

    aa = dq[0]

    while seqs:
        # find the next closest point on each end
        ind = numpy.argmin([similarity_metric(p=aa, q=s) for s in seqs])
        dq.append(seqs[ind])
        aa = seqs[ind]  # tuple to search next
        seqs.pop(ind)
        # print(dq)
        
    dd = {}  # vocab with key = sequence and value = index
    cntr = 0
    for d in dq:
        dd[d] = cntr
        cntr += 1

    return dd

ldat = lottery_data()
vocab, vocab_size = vocabulary_helper3(i_min=1, i_max=31, j_min=2, j_max=32, k_min=3, k_max=33)
print(f"vocab size: {vocab_size}")
vocab_sim = vocabulary_tokens_by_similarity(vocab=vocab)
assert len(vocab_sim) == len(vocab)
vocab_file = Path.cwd().joinpath('data','vocab.pkl')
with open(vocab_file, 'wb') as fobj:
    pickle.dump(vocab_sim, fobj)
    fobj.close()
