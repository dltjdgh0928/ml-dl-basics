import numpy as np
a = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
a = np.transpose(a)
print(a.shape)
def outliers(a):
    b=[]
    for i in range(a.shape[1]):
        q1, q2, q3 = np.percentile(a[:, i], [25, 50, 75], axis=0)
        print('q1 : ', q1)
        print('q2 : ', q2)
        print('q3 : ', q3)
        iqr = q3 - q1
        print('iqr : ', iqr)
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        b.append(np.where((a[:,i]>upper_bound)|(a[:,i]<lower_bound)))
    return b

outliers_loc = outliers(a)    
print('이상치의 위치 : ', outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(outliers_loc)
# plt.show()