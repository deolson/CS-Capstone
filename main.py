import network_model

def main():
    #time layer #of nuerouns/ #of timeLayers/ #of noteLayer neurons/ #of noteLayers/ dropout
    model = network_model.choraleModel(True,[300,300],2,[100,50],2,0.5)

if __name__ == '__main__':
    main()
