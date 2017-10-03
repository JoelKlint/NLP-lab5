"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import features

import time
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer

def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra'# + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la'# + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))
    """
    Results in:
    ['B-ADJP', 'B-ADVP', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-NP', 'B-PP',
    'B-PRT', 'B-SBAR', 'B-UCP', 'B-VP', 'I-ADJP', 'I-ADVP', 'I-CONJP',
    'I-INTJ', 'I-NP', 'I-PP', 'I-PRT', 'I-SBAR', 'I-UCP', 'I-VP', 'O']
    """
    # We assign each name a number
    dict_classes = dict(enumerate(classes))
    """
    Results in:
    {0: 'B-ADJP', 1: 'B-ADVP', 2: 'B-CONJP', 3: 'B-INTJ', 4: 'B-LST',
    5: 'B-NP', 6: 'B-PP', 7: 'B-PRT', 8: 'B-SBAR', 9: 'B-UCP', 10: 'B-VP',
    11: 'I-ADJP', 12: 'I-ADVP', 13: 'I-CONJP', 14: 'I-INTJ',
    15: 'I-NP', 16: 'I-PP', 17: 'I-PRT', 18: 'I-SBAR',
    19: 'I-UCP', 20: 'I-VP', 21: 'O'}
    """

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}
    """
    Results in:
    {'B-SBAR': 8, 'I-NP': 15, 'B-PP': 6, 'I-SBAR': 18, 'I-PP': 16, 'I-ADVP': 12,
    'I-INTJ': 14, 'I-PRT': 17, 'I-CONJP': 13, 'B-ADJP': 0, 'O': 21,
    'B-VP': 10, 'B-PRT': 7, 'B-ADVP': 1, 'B-LST': 4, 'I-UCP': 19,
    'I-VP': 20, 'B-NP': 5, 'I-ADJP': 11, 'B-CONJP': 2, 'B-INTJ': 3, 'B-UCP': 9}
    """

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes


if __name__ == '__main__':
    train_file = 'corpus/sv/swedish_talbanken05_train.conll'
    test_file = 'corpus/sv/swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    feature_names_short = [
        'stack_0_word',
        'stack_0_POS',
        'queue_0_word',
        'queue_0_POS',
        'can-re',
        'can-la'
    ]

    feature_names_middle = [
        'stack_0_word',
        'stack_0_POS',
        'stack_1_word',
        'stack_1_POS',
        'queue_0_word',
        'queue_0_POS',
        'queue_1_word',
        'queue_1_POS',
        'can-re',
        'can-la'
    ]

    feature_names_long = [
        'stack_0_POS',
        'stack_0_word',
        'stack_1_word',
        'stack_1_POS',
        'queue_0_word',
        'queue_0_POS',
        'after_stack_0_word',
        'after_stack_0_POS',
        'can-re',
        'can-la'
        'can-ra',
        'root_of_sentence'
    ]

    sent_cnt = 0

    y_symbols = [] # Our array of transistions
    X_dict = list() # Our matrix

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            a = 1
            # print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'


        while queue:
            x = features.extract(stack, queue, graph, feature_names_long, sentence)
            X_dict.append(x)

            stack, queue, graph, trans = reference(stack, queue, graph)
            
            y_symbols.append(trans)
        stack, graph = transition.empty_stack(stack, graph)



        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]
        # print(y_symbols)
        # print(graph)


    """
        Start SKlearn part
    """
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    y, dict_classes, inv_dict_classes = encode_classes(y_symbols)

    print("Training the model...")
    training_start_time = time.clock()
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear', verbose=1)
    # classifier = linear_model.Perceptron(penalty='l2')
    # classifier = tree.DecisionTreeClassifier()
    model = classifier.fit(X, y)
    print(model)

    y_train_predicted = model.predict(X)

    # TODO: Test the model with test data

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_symbols, list(map(lambda y_pred: dict_classes[y_pred], y_train_predicted)) )))

