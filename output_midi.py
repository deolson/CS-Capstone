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

    events = StateMatrix[key[0]]
    for i in range(len(events)):
        # print("---- %i ----" %(i))
        for j in range(len(events[i])):

            # [1, 1] = Note started
            if events[i][j] == [1, 1]:
                on = midi.NoteOnEvent(tick=i, velocity=90, pitch=j)
                track.append(on)
                # print("midi.%s" %(str(getNote(j))))


            # [1, 0] = note held over from before
            # if events[i][j] == [1, 0]:
            #     print(getNote(j))

            # [0, 0] = note not playing
            if events[i][j] == [0, 0] and events[i - 1][j] != [0, 0]:
                off = midi.NoteOffEvent(tick=i, pitch=j)
                track.append(off)
                # print("%s off" %(getNote(j)))

    eot = midi.EndOfTrackEvent(tick=len(events))
    track.append(eot)
    print pattern
    midi.write_midifile("output.mid", pattern)


def getNote(intNote):
    oct = intNote/12
    letter = switch(intNote%12)
    return letter+str(oct)

def switch(x):
    return {
        0 : 'C_',
        1 : 'C#_',
        2 : 'D_',
        3 : 'D#_',
        4 : 'E_',
        5 : 'F_',
        6 : 'F#_',
        7 : 'G_',
        8 : 'G#_',
        9 : 'A_',
        10 : 'A#_',
        11 : 'B_',
    }[x]




if __name__ == '__main__':
    DataFile = open("dataJSON.json", "rb")
    StateMatrix = json.load(DataFile)
    DataFile.close()
    StateMatrixtoMidi(StateMatrix)
