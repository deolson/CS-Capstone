# state matrix returns all data in a dataset into a hashmap of state matrices
# need a way to return a state matrix back into a midi file
# using python 2.7 as a script, takes hashmap structure and writes it to data.txt,
#   which is then ported into main.py and interpreted in 3.x
import numpy
import midi
import cPickle
import os


def StateMatrixtoMidi(StateMatrix):
    key = StateMatrix.keys()
    events = StateMatrix[key[0]]

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    for i in range(len(events)):
        if events[i] != [0, 0, "A4"]:
            print(i)



if __name__ == '__main__':
    DataFile = open("data.txt", "rb")
    StateMatrix = cPickle.load(DataFile)
    DataFile.close()
    # speckey = StateMatrix.keys()
    # print(speckey)
    # print(StateMatrix[speckey[0]][20])
    StateMatrixtoMidi(StateMatrix)
