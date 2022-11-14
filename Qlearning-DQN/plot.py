import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dq = dq = np.load('dq.npy')

plt.figure()
plt.plot(dq, alpha=0.5, label='Raw')
plt.plot(pd.DataFrame(dq).rolling(10).mean(), c='b', label='MA 10')
plt.hlines(0, 0, 1000, color='k')
plt.xlabel('Episode')
plt.ylabel('Q change')
plt.xlim([0, 1000])
plt.title('Mean Q table change per episode')
plt.legend()
plt.savefig('1.jpeg')
