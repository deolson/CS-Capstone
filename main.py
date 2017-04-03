import pickle
import network_model
import train_model


def main():
    datafile = open( "data.txt", "rb" )
    matDict = pickle.load(datafile);
    datafile.close()

    #time layer #of nuerouns/ #of timeLayers/ #of noteLayer nuerouns/ #of noteLayers/ dropout
    model = network_model.choraleModel([300,300],2,[100,50],2,0.5)

if __name__ == '__main__':
    main()
