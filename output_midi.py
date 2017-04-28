# # state matrix returns all data in a dataset into a hashmap of state matrices
# # need a way to return a state matrix back into a midi file
# # using python 2.7 as a script, takes hashmap structure and writes it to data.txt,
# #   which is then ported into main.py and interpreted in 3.x
import numpy
import midi
import json
import os
import time as timet

# def DictToStateMatrix(matDict):
#     keys = matDict.keys()
#
#     for key in keys:
#         StateMatrix = matDict.get(key)
#         StateMatrixtoMidi(StateMatrix)

def StateMatrixtoMidi(StateMatrix):
    key = StateMatrix.keys()
    StateMatrix = numpy.asarray(StateMatrix[key[0]])

    total_notes = 128
    tick_scale = 55

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)


    # StateMatrix[key[0]], StateMatrix[key[0]][0], and StateMatrix[key[0]][0][0]
    # are lists
    # StateMatrix[key[0]][0][0][0] is an int

    lastcmdtime = 0

    prev_state = [[0,0] for x in range(total_notes)]

    for time, state in enumerate(StateMatrix + prev_state):
        off_notes = []
        on_notes = []

        for i in range(total_notes):
            if prev_state[i][0] == 1:

                if state[i][0] == 0:
                    off_notes.append(i)

                elif state[i][1] == 1:
                    off_notes.append(i)
                    on_notes.append(i)

            elif state[i][0] == 1:
                on_notes.append(i)

        for note in off_notes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tick_scale, pitch = note))
            lastcmdtime = time

        for note in on_notes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tick_scale, velocity = 40, pitch = note))
            lastcmdtime = time

        prev_state = state

    eot = midi.EndOfTrackEvent(tick=len(StateMatrix))
    track.append(eot)
    # print pattern

    localtime   = timet.localtime()
    timeString  = timet.strftime("%Y%m%d%H%M%S", localtime)
    name = timeString+".mid"
    midi.write_midifile(name, pattern)
    os.rename("./%s" % (name), "./Midis/%s" % (name))


if __name__ == '__main__':
    DataFile = open("dataJSON.json", "r")
    StateMatrix = json.load(DataFile)
    DataFile.close()
    StateMatrixtoMidi(StateMatrix)
