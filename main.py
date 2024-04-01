import RUS_classes as rus
import sys
import numpy as np

if __name__ == "__main__":
    num = int(sys.stdin.readline().strip())
    for _ in range(num):
        N = int(sys.stdin.readline().strip())
        type = sys.stdin.readline().strip()
        shape = sys.stdin.readline().strip()
        system = sys.stdin.readline().strip()
        limits = np.array(list(map(float,sys.stdin.readline().strip().split())))
        rho = float(sys.stdin.readline().strip())
        C = np.zeros((6,6))
        for r in range(6):
            row = list(map(float,sys.stdin.readline().strip().split()))
            for l in range(len(row)):
                C[r,l] = row[l]
        Forward = rus.Forward(N,type,shape,system,limits,rho,C)
        w2 = Forward.W2
        for w in range(len(w2)):
            sys.stdout.write(str(w2[w]))
            sys.stdout.write("\n")
