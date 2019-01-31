import numpy as np

def isInRect(x,in_a):
    a = in_a*0.0
    a[0] = in_a[0]
    d = in_a - in_a[0]
    tan = np.arctan2(d[1:,1],d[1:,0])
    args = np.argsort(tan)
    a[1:] = in_a[1:][args]

    a1 = abs(np.cross(x-a[0],x-a[1])*0.5)
    a2 = abs(np.cross(x-a[1],x-a[2])*0.5)
    a3 = abs(np.cross(x-a[2],x-a[3])*0.5)
    a4 = abs(np.cross(x-a[3],x-a[0])*0.5)

    A1 = abs(np.cross(a[1]-a[0],a[3]-a[0])*0.5)
    A2 = abs(np.cross(a[1]-a[2],a[3]-a[2])*0.5)

    if A1+A2 == a1+a2+a3+a4:
        return True
    return False
    print(a1,a2,a3,a4,a1+a2+a3+a4)
    print(A1,A2,A1+A2)
    exit()



if __name__ == '__main__':
    r = np.array([[0.0,0.0],[1.0,1.2],[0.0,1.0],[1.0,0.0]])
    m = np.array([0,10])
    print(isInRect(m,r))
