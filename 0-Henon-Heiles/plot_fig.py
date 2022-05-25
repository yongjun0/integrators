'''
'''
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys

order1=[60, 61, 62, 63, 64]
d_folder = "data"

#plt.figure()
for i in range( len(order1)):
    file_name = d_folder+'/'+'err_'+str(order1[i])+'.npz'
    data = np.load(file_name)
    dt= data['dt']
    err = data['err']
    plt.loglog(dt, err)

    x1 = np.log10(dt[0])
    x2 = np.log10(dt[-1])
    y1 = np.log10(err[0])
    y2 = np.log10(err[-1])
    m = (y2-y1)/(x2-x1)
    print(order1[i], m)

#plt.show()
#print(len(t), len(x_an), len(p_an), len(H_an))
#sys.exit()

if(0):
    plt.figure()
    plt.subplot(311)
    plt.plot(t, x_an, 'k', label = "Analytic")
    plt.plot(t, x_SE, 'r', label = "SE")
    #plt.plot(t, x_TE, 'k-o', label = "Taylor")
    plt.plot(t, x_VV, 'g', label = "VV")
    #plt.plot(t, x_OV, 'b', label = "OV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.subplot(312)
    plt.plot(t, p_an, 'k', label = "Analytic")
    plt.plot(t, p_SE, 'r', label = "SE")
    #plt.plot(t, x_TE, 'k-o', label = "Taylor")
    plt.plot(t, p_VV, 'g', label = "VV")
    #plt.plot(t, x_OV, 'b', label = "OV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("V")
    plt.subplot(313)
    plt.hlines(y = H_an, xmin=0, xmax=T, label = "Analytic")
    plt.plot(t, H_SE, 'r', label = "SE")
    plt.plot(t, H_SE+err_SE, 'r:', label = "Shadow H_SE")
    plt.plot(t, H_VV, 'g', label = "VV")
    plt.plot(t, H_VV+err_VV, 'g:', label = "Shadow H_VV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("H")
    plt.suptitle(f'dt = {h:.2f}')





plt.show()


