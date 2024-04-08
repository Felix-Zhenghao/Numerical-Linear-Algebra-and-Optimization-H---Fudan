import numpy as np
import matplotlib.pyplot as plt

# Read the file
with open('rjj.txt', 'r') as file:
    data = file.read()

# Convert the string to a list of floats
data_list = list(map(float, data.split()))
# Convert the list to a numpy array and take log10
rjj = np.log10(np.array(data_list))
# Split the list into three parts
CGS_rjj = rjj[:int(len(rjj)/3)]
MGS_rjj = rjj[int(len(rjj)/3):int(2*len(rjj)/3)]
HQR_rjj = rjj[int(2*len(rjj)/3):]
j = np.arange(1, len(CGS_rjj)+1)


# Plot the rjj verses j in logy style in one figure
# j is the index of the rjj and shoul be integer
plt.figure()
plt.plot(j, CGS_rjj, label='CGS')
plt.plot(j, MGS_rjj, label='MGS')
plt.plot(j, HQR_rjj, label='HQR')
plt.legend()
plt.show()


