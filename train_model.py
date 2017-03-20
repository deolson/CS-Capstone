def getBatchPieces(matDict):
    pass
    
def trainModel(model,matDict,trainingIterations):
    for _ in range(trainingIterations):
        print(getBatchPieces(matDict))
