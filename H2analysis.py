import pandas as pd
df1 = pd.read_csv('Contrast1.csv')
df2 = pd.read_csv('ContextQs2.csv')
# So going to be looking at df2['Average'] (Av Experience for each subject) and df1['value'] (contrast values for each subject)

# Check for normality - contrast values normality checked already for H1
# For AvExperience - histogram and then Shapiro Wilk
import matplotlib.pyplot as plt
plt.figure(1) #this important to esnure separate from future plots
df2['Average'].plot(kind='hist')
plt.title('Histogram of Average Visual Experience')
plt.xlabel('Average Visual Experience')
plt.ylabel('Frequency')
plt.savefig('AverageContext.png')
from scipy import stats
print(stats.shapiro(df2['Average']))

#Scatterplot comparing the two variables
import numpy as np
plt.figure(2)
plt.scatter(df2['Average'], df1['value'])
m, b = np.polyfit(df2['Average'], df1['value'], 1)
plt.plot(df2['Average'], m*(df2['Average'])+b, color='grey')
plt.xlabel('Mean Visual Experience Rating')
plt.ylabel('Contrast Value')
plt.savefig('AverageContextContrast.png')
print(50)

# Pearson's r
res = stats.pearsonr(df2['Average'], df1['value'])
print(res)
print(res.confidence_interval(confidence_level=0.95))
print(100)