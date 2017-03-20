import pickle

def main():
    datafile = open( "data.txt", "rb" )
    matDict = pickle.load(datafile);
    datafile.close()
    print(matDict.keys())

if __name__ == '__main__':
    main()
