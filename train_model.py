import random
batch_len = 16*8 #of segments trained on in 16th notes 16*8 = 8 measures -- training speed vs time patterns
division_len = 16 #step size of measures, we dont want to start a batch in the middle of a measure curr in 16th notes
batch_width = 10



def context():
    pass

def beat():
    pass

def noteStateSingleToInputForm(time,state):
    pass


def batchToVector(statematrixBatch):
    inputform = [noteStateSingleToInputForm(state,time) for time,state in enumerate(statematrixBatch)]
    return inputform

def getBatchPieces(matDict):
    #grab a random statematrix
    randomStatematrix = random.choice(list(matDict.values()))
    #where to start the sample in our statematrix, 0->(end-how big the batch is), only chooseing values at he start of measures
    startindx = random.randrange(0,len(randomStatematrix)-batch_len,division_len)
    batch = randomStatematrix[startindx:startindx+batch_len] #take the batch from the random statematrix
    modelInput = batchToVector(batch)
    print(batch)

def trainModel(model,matDict,trainingIterations):
    # for _ in range(trainingIterations):
        print(getBatchPieces(matDict))
