from utils import *

def main():
    S = GetSequences()
    print(S.shape)
    for i in S:
        print(i)
        break

if __name__ == '__main__':
    main()