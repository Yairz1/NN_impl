def accuracy(prediction, labels):
    correct = 0
    for p1, p2 in zip(prediction, labels):
        if p1 == p2: correct += 1
    return correct / labels.shape[1]
