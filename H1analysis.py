import pandas as pd

df = pd.read_csv('Contrast1.csv')

#mean of contrasts
import statistics
mean = statistics.mean(df['value'])
SD = statistics.stdev(df['value'])

# Histogram of contrast values
import matplotlib.pyplot as plt
plt.hist(df['value'], bins=20)
plt.xlabel('Contrast Values')
plt.ylabel('Frequency')
plt.savefig('Contrast1Hist.png')

#Test for normality
from scipy import stats
print(stats.shapiro(df['value']))

#One sample T-test to assess whether contrast values greater than 0 - ie whether leading diagonal greater than off diagonal
print(stats.ttest_1samp(df['value'], 0))

# calculating Cohen's d
# d = (sample mean - 0)/ sample sd with n-1 dof
d = mean/SD
print(d)