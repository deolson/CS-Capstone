# state matrix returns all data in a dataset into a hashmap of state matrices
# need a way to return a state matrix back into a midi file
# using python 2.7 as a script, takes hashmap structure and writes it to data.txt,
#   which is then ported into main.py and interpreted in 3.x
import numpy
import midi
import json
import os


def StateMatrixtoMidi(StateMatrix):
    key = StateMatrix.keys()

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    # StateMatrix[key[0]], StateMatrix[key[0]][0], and StateMatrix[key[0]][0][0]
    # are lists
    # StateMatrix[key[0]][0][0][0] is an int

    #gets all actual events from StateMatrix
    for i in range(len(StateMatrix[key[0]])):
        for j in range(len(StateMatrix[key[0]][i])):
            if 1 in StateMatrix[key[0]][i][j]:
                print(StateMatrix[key[0]][i][j])







if __name__ == '__main__':
    DataFile = open("dataJSON.json", "rb")
    StateMatrix = json.load(DataFile)
    DataFile.close()
    # speckey = StateMatrix.keys()
    # print(speckey)
    # print(StateMatrix[speckey[0]][20])
    StateMatrixtoMidi(StateMatrix)
