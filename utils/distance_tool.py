import math
def Euclidean_fun(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))