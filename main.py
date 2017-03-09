import os, sys
import statematrix

def main():
    path = "./OneDataSet"
    dirs = os.listdir( path )
    for file in dirs:
        print(file)

if __name__ == '__main__':
    main()
