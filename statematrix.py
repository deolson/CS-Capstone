import numpy
import midi
import cPickle
import os

totalNotes = 128;

def midiToStatematrix(midifile):
    pattern = midi.read_midifile(midifile)
    pattern.make_ticks_abs()

    statematrix = []
    currNotes = [[0,0,getNote(x)] for x in range(totalNotes)]
    trackPositions = [track[0].tick for track in pattern]

    resolution = pattern.resolution
    pulse = 0

    while True:
        if pulse % resolution/2 == 0:
            oldNotes = currNotes[:]
            statematrix.append(oldNotes)
        if pulse % resolution/2 == resolution/4:
            oldNotes = currNotes[:]
            statematrix.append(oldNotes)

        for i in range(len(pattern)):
            # print trackPositions
            # print "track position"+str(trackPositions)
            # print "pulse"+str(pulse)
            # print "i"+str(i)
                # continue
            while pattern[i][trackPositions[i]].tick == pulse:
                track = pattern[i]
                position = trackPositions[i]
                event = track[position]

                if midi.NoteOnEvent.is_event(event.statusmsg):
                    currNotes[event.pitch] = [1,1,getNote(event.pitch)]
                if midi.NoteOffEvent.is_event(event.statusmsg):
                    currNotes[event.pitch] = [0,0,getNote(event.pitch)]


                trackPositions[i] += 1
                if trackPositions[i] >= len(track):
                    trackPositions[i] = 0
                    break

        pulse += 1
        if tracksDone(trackPositions):
            break
    # for array in statematrix:
    #     printCurrPlaying(array)
    #     print("----------------------------")
    return statematrix

def tracksDone(trackPositions):
    for position in trackPositions:
        if position != 0:
            return False
    return True

def getNote(intNote):
    oct = intNote/12
    letter = switch(intNote%12)
    return letter+str(oct)

def printCurrPlaying(currNotes):
    for array in currNotes:
        if array[0] == 1 or array[1] == 1:
            print(array[2])

def switch(x):
    return {
        0 : 'C',
        1 : 'C#',
        2 : 'D',
        3 : 'D#',
        4 : 'E',
        5 : 'F',
        6 : 'F#',
        7 : 'G',
        8 : 'G#',
        9 : 'A',
        10 : 'A#',
        11 : 'B',
    }[x]

def filesToDict(path):
    return matDict

if __name__ == '__main__':
    #dict is essentially python equiv of a hashmap
    path = "./OneDataSet/"
    matDict = {}
    dirs = os.listdir(path)
    for file in dirs:
        matDict[path+str(file)] = midiToStatematrix(path+str(file));
    # print(matDict)
    dataFile = open('data.txt', 'wb')
    cPickle.dump(matDict, dataFile, cPickle.HIGHEST_PROTOCOL)
    dataFile.close()
