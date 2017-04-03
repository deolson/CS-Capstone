import random
import numpy
from main import matDict
numpy.set_printoptions(threshold=numpy.nan)

batch_len = 2#16*8 #of segments trained on in 16th notes 16*8 = 8 measures -- training speed vs time patterns
division_len = 16 #step size of measures, we dont want to start a batch in the middle of a measure curr in 16th notes
binary_len = 4 # number of bits needed to rep division_len
batch_width = 2

def getWithinAnOct(state, note):
    withinOct = []
    for i in range(-12, 13):
        try:
            withinOct += state[note+i]
        except IndexError:
            withinOct += [0,0]
    return withinOct

def getSingleVector(note, state, beat, context):
    position = note % 12

    pitchclass = [0]*12
    pitchclass[position] = 1

    currNoteContext = context[position:] + context[:position]

    withinAnOct = getWithinAnOct(state,note)

    return [note] + pitchclass + withinAnOct + currNoteContext + beat +[0]


def getContext(state):
    context = [0] *12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = note % 12
            context[pitchclass] += 1
    return context

def getBeat(time):
    binary = [int(i) for i in '{0:04b}'.format((time % division_len))]
    for i in range(len(binary)):
        if binary[i] == 0:
            binary[i] = -1
    return binary

def stateToInputVectorArray(time,state):
    beat = getBeat(time)
    context = getContext(state)
    vectors = [getSingleVector(note,state,beat,context) for note in range(len(state))]
    return vectors


def batchToVectors(statematrixBatch):
    inputform = [stateToInputVectorArray(time,state) for time,state in enumerate(statematrixBatch)]
    return inputform

def getBatchPieces(matDict):
    #grab a random statematrix
    randomStatematrix = random.choice(list(matDict.values()))
    #where to start the sample in our statematrix, 0->(end-how big the batch is), only chooseing values at he start of measures
    startindx = random.randrange(0,len(randomStatematrix)-batch_len,division_len)
    batch = randomStatematrix[startindx:startindx+batch_len] #take the batch from the random statematrix
    modelInput = batchToVectors(batch)
    return batch, modelInput

def getModelInputs():
    batch, modelInput = zip(*[getBatchPieces(matDict) for _ in range(batch_width)])
    batch = numpy.array(batch)

    modelInput = numpy.array(modelInput)
    modelInput = modelInput[:,0:-1]
    tensorShapeModelInput = modelInput.transpose(1,0,2,3).reshape(batch_len-1,batch_width*128,80)
    return batch, tensorShapeModelInput
