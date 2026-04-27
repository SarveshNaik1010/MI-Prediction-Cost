# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel("medical_insurance_cost_prediction_dataset.xlsx")

# %%
# UNIVARIATE ANALYSIS

# %%
plt.figure()
plt.hist(df['Insurance Cost'], bins=15)
plt.title("Histogram of Insurance Cost")
plt.xlabel("Insurance Cost")
plt.ylabel("Frequency")
plt.show()

# %%
plt.figure()
plt.hist(df['Age'], bins=15)
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# %%
plt.figure()
gender_counts = df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# %%
plt.figure()
smoker_counts = df['Smoker'].value_counts()
plt.pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%')
plt.title("Smoking Status Distribution")
plt.show()

# %%
plt.figure()
df['Age'].plot(kind='density')
plt.title("Density Plot of Age")
plt.xlabel("Age")
plt.show()

# %%
plt.figure()
plt.boxplot(df['Annual Income'])
plt.title("Boxplot of Annual Income")
plt.ylabel("Annual Income")
plt.show()

# %%
# BIVARIATE ANALYSIS

# %%
plt.figure()
plt.scatter(df['Age'], df['Insurance Cost'])
plt.title("Age vs Insurance Cost")
plt.xlabel("Age")
plt.ylabel("Insurance Cost")
plt.show()


# %%
plt.figure()
children_avg = df.groupby('Children')['Insurance Cost'].mean()
plt.bar(children_avg.index, children_avg.values)
plt.title("Average Insurance Cost by Number of Children")
plt.xlabel("Children")
plt.ylabel("Average Cost")
plt.show()

# %%
plt.figure()
sorted_df = df.sort_values(by='Age')
plt.plot(sorted_df['Age'], sorted_df['Insurance Cost'])
plt.title("Line Plot: Age vs Insurance Cost")
plt.xlabel("Age")
plt.ylabel("Insurance Cost")
plt.show()

# %%
plt.figure()
data = [
    df[df['Smoker']=='Yes']['Insurance Cost'],
    df[df['Smoker']=='No']['Insurance Cost']
]
plt.boxplot(data, labels=['Smoker','Non-Smoker'])
plt.title("Insurance Cost by Smoking Status")
plt.ylabel("Insurance Cost")
plt.show()

# %%
plt.figure()
plt.scatter(df['Annual Income'], df['Insurance Cost'])
plt.title("Annual Income vs Insurance Cost")
plt.xlabel("Annual Income")
plt.ylabel("Insurance Cost")
plt.show()

# %%
# MULTIVARIATE ANALYSIS

# %%
plt.figure()
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['BMI'], df['Insurance Cost'])
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Insurance Cost')
ax.set_title("3D Scatter: Age, BMI, Insurance Cost")
plt.show()

# %%
plt.figure()
grouped = df.groupby(['Gender','Smoker'])['Insurance Cost'].mean().unstack()
grouped.plot(kind='bar')
plt.title("Average Insurance Cost by Gender and Smoking")
plt.ylabel("Average Cost")
plt.show()

# %%
plt.figure()
df['Age Group'] = pd.cut(df['Age'], bins=5)
multi_line = df.groupby(['Age Group','Smoker'])['Insurance Cost'].mean().unstack()
multi_line.plot()
plt.title("Average Cost by Age Group and Smoking")
plt.ylabel("Average Cost")
plt.show()

# %%
plt.figure()
pivot = df.pivot_table(values='Insurance Cost',
                       index='Region',
                       columns='Policy Type',
                       aggfunc='mean')
plt.imshow(pivot)
plt.colorbar()
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Heatmap: Region vs Policy Type (Avg Cost)")
plt.show()

# %%
from pandas.plotting import scatter_matrix

plt.figure()
numeric_df = df.select_dtypes(include='number')
scatter_matrix(numeric_df, figsize=(10,10))
plt.show()

# %%



