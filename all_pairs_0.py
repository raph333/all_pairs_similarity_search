#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple demo for understanding the basic inverted list approach of google's
all-pairs algorithm (termed ALL-PAIRS-0 in the paper).
Get the paper here: https://static.googleusercontent.com/media/
research.google.com/en//pubs/archive/32781.pdf

A mini-text file of five sentences is used in order to track each step of the
algorithm. For better understanding, everything is implemted from scratch.
"""

import numpy as np


# =============================================================================
# def jaccard_sim(r,s):
#     intersect = set(r).intersection(set(s))
#     union = set(r).union(set(s))
#     similarity = len(intersect) / len(union)
#     return similarity
# =============================================================================


def read_txt_file(filename):
    """
    @ filename: path to text file
    Reads each into a list of words.
    return: list of list of words
    """
    data_list = []
    with open(filename) as infile:
        for line in infile.readlines():
            data_list.append([x.strip('.,;') for x in line.split()])
    return data_list


# basic vector operations:
# =============================================================================
def cardinality(vector):
    return np.sqrt(sum([x**2 for x in vector]))


def dot_product(listA, listB):
    return sum([a*b for a,b in zip(listA,listB)])


def cosine_sim(listA, listB):
    return dot_product(listA, listB) / (cardinality(listA)*cardinality(listB))
# =============================================================================


# functions for document vectors:
# =============================================================================
def make_document_vectors(ds):
    char_set = set.union(*[set(x) for x in ds])
    document_vectors = []
    for vector in ds:
        document_vect = [vector.count(x) for x in char_set]
        document_vectors.append(document_vect)
    return document_vectors


def normalize_doc_vectors(doc_vectors):
    normalized_vectors = []
    for vector in doc_vectors:
        norm_vect = [x/cardinality(vector) for x in vector]
        normalized_vectors.append(norm_vect)
    return normalized_vectors
        

def sparse_representation(document_vectors):
    sparse_vectors = []
    for vector in document_vectors:
        sparse_vect = [x for x in enumerate(vector) if x[1] > 0]
        sparse_vectors.append(sparse_vect)
    return sparse_vectors
# =============================================================================
    

def check_results(res, document_vectors):
    """
    @ res: similar pairs: (vector_ID1, vector_ID2, similarity)
    @ document_vectors: list of original document vectors
    purpose: Check all-pairs implementation (which uses normalized document
    vectors in sparse representation and gradually builds up to cosine
    similarity)
    return: checked pairs: (vector_ID1, vector_ID2, True/False)
    """
    checked_pairs = []
    for pair in res:
        vect1 = document_vectors[pair[0]]
        vect2 = document_vectors[pair[1]]
        is_correct = round(pair[2], 4) == round(cosine_sim(vect1, vect2), 4)
        checked_pairs.append( (pair[0], pair[1], is_correct) )
    return checked_pairs


def get_dimensionality(V):
    """
    @ V: list of vectors in sparse representation
    return: dimension of vectors (m)
    """
    flat_list = [x for vector in V for x in vector]
    dimensions = [x[0] for x in flat_list]
    return max(dimensions) + 1


def find_matches(x_ID, x, I, t):
    """
    @ x_ID: just an identifier for vector x element of V
    @ x : vector x in sparse representation
    @ I: inverted list (as dict)
    @ t: similarity threshold
    """
    A = {}  # vectorID2weight
    M = []
    
    for i, xi in x:   # (for each i s.t. x[i] > 0)
        for y_ID, yi in I[i]:
            if y_ID not in A.keys():  # *
                A[y_ID] = 0
            A[y_ID] = A[y_ID] + xi*yi
    
    for y_ID in A.keys():  # only non-0 keys are initialized * 
        if A[y_ID] > t:
            M.append( (x_ID, y_ID, A[y_ID]) )
    return M


def all_pairs0(V, t):
    """
    @ V: list of vectors in sparse representation
    @ t: similarity threshold
    """
    m = get_dimensionality(V)  # dimension of vectors in V
    print('dimension m of vectors in V: %s' % m)
    O = []  # list of similar pairs (results)
    I = {k:[] for k in range(m)}  # empty inverted list
    
    for x_ID in range(len(V)):
        x = V[x_ID]
        print('\nvector number %s:\n%s' % (x_ID, x))
        
        matches = find_matches(x_ID, x, I, t)
        print('found matches: %s' % matches)
        O = O + matches
        
        for i, xi in x:   # (for each i s.t. x[i] > 0)
            I[i] = I[i] + [(x_ID, xi)]
        print('updated I:\n%s' % {k:v for k,v in I.items() if v})
    
    return O


if __name__ == '__main__':

    mini = read_txt_file('mini_data_set.txt')
    doc_vectors = make_document_vectors(mini)
    norm_doc_vectors = normalize_doc_vectors(doc_vectors)
    sparse_vectors = sparse_representation(norm_doc_vectors)
    
    result = all_pairs0(sparse_vectors, 0.5)
    print('\n\nresult: %s matching pairs:\n%s' % (len(result), result))
    
    checked_result = check_results(result, doc_vectors)
    print('\nchecking pairs:\n%s' % checked_result)

        