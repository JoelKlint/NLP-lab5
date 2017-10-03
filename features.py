import transition

# TODO: Should we not use sentence?
def extract(stack, queue, graph, feature_names, sentence):
    """
    Returns a row
    """

    features = list()

    # TODO: Should we use postag os cpostag?
    POS_TAG = 'postag'
    WORD_TAG = 'form'
    ID_TAG = 'id'
    DEPREL_TAG = 'deprel'
    NULL_VALUE = 'nil'

    # stack_0
    if stack:
        stack_0_POS = stack[0][POS_TAG]
        stack_0_word = stack[0][WORD_TAG]
    else:
        stack_0_POS = NULL_VALUE
        stack_0_word = NULL_VALUE

    # stack_1
    if len(stack) > 1:
        stack_1_POS = stack[1][POS_TAG]
        stack_1_word = stack[1][WORD_TAG]
    else:
        stack_1_POS = NULL_VALUE
        stack_1_word = NULL_VALUE

    # queue_0
    if queue:
        queue_0_POS = queue[0][POS_TAG]
        queue_0_word = queue[0][WORD_TAG]
    else:
        queue_0_POS = NULL_VALUE
        queue_0_word = NULL_VALUE

    # queue_1
    if len(queue) > 1:
        queue_1_POS = queue[1][POS_TAG]
        queue_1_word = queue[1][WORD_TAG]
    else:
        queue_1_POS = NULL_VALUE
        queue_1_word = NULL_VALUE

    # word after top of stack in sentence
    if stack_0_word == NULL_VALUE:
        after_stack_0_word = NULL_VALUE
        after_stack_0_POS = NULL_VALUE
    else:
        id_stack_0 = int(stack[0]['id'])
        if len(sentence)-1 == id_stack_0: #stack 0 is the last word
            after_stack_0_word = NULL_VALUE
            after_stack_0_POS = NULL_VALUE
        else:
            next_word = sentence[id_stack_0+1]
            after_stack_0_word = next_word[WORD_TAG]
            after_stack_0_POS = next_word[POS_TAG]
    
    # The root word of the sentence
    for word in sentence:
        if word[DEPREL_TAG] == 'ROOT' and word[POS_TAG] != 'ROOT':
            root_word_of_sentence = word[WORD_TAG]
            break

    if len(feature_names) == 6:
        features.append(stack_0_word)
        features.append(stack_0_POS)

        features.append(queue_0_word)
        features.append(queue_0_POS)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    elif len(feature_names) == 10:
        features.append(stack_0_word)
        features.append(stack_0_POS)

        features.append(stack_1_word)
        features.append(stack_1_POS)

        features.append(queue_0_word)
        features.append(queue_0_POS)

        features.append(queue_1_word)
        features.append(queue_1_POS)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

    elif len(feature_names) == 14:
        features.append(stack_0_word)
        features.append(stack_0_POS)

        features.append(stack_1_word)
        features.append(stack_1_POS)

        features.append(queue_0_word)
        features.append(queue_0_POS)

        features.append(queue_1_word)
        features.append(queue_1_POS)

        features.append(after_stack_0_word)
        features.append(after_stack_0_POS)

        features.append(transition.can_reduce(stack, graph))
        features.append(transition.can_leftarc(stack, graph))

        # Our own features
        features.append(transition.can_rightarc(stack, graph))
        features.append(root_word_of_sentence)


    # Convert features object
    features = dict(zip(feature_names, features))


    return features