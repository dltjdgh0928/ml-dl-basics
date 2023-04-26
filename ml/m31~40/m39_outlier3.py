import numpy as np
a = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).transpose()

def outliers(a):
    b = []
    for i in range(a.shape[1]):
        q1, q2, q3 = np.percentile(a[:, i], [25, 50, 75], axis=0)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - (iqr * 1.5), q3 + (iqr * 1.5)
        print(f'q1 :  {q1}\n q2 : {q2}\n q3 : {q3}\n iqr : {iqr}\n')
        b.append(np.where((a[:,i]>upper_bound)|(a[:,i]<lower_bound)))
    return b

print('location of outliers : ', outliers(a))