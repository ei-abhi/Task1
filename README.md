
# 🚢 Titanic Dataset Analysis - Task 1

This notebook performs a basic data cleaning and preprocessing pipeline on the Titanic dataset, including handling missing values, encoding categorical variables, outlier detection, and visualization.

---

## 📁 Dataset

The dataset used is: `Titanic-Dataset.csv`  
It contains information about passengers aboard the Titanic, including their age, gender, class, and survival status.

---

## 📌 Workflow Summary

### 1. **Data Import and Exploration**

```python
import pandas as pd
df = pd.read_csv('/content/Titanic-Dataset.csv')
df.head()
df.isnull().sum()
df.dtypes
df.info()
df.describe()
df.columns
```

- **`df.head()`** shows the first 5 rows.
- **`isnull().sum()`** checks missing values in each column.
- **`info()` and `describe()`** give an overview of data types and statistics.

---

### 2. **Handling Missing Values in 'Age'**

```python
df['Age'].fillna(df['Age'].mode()[0], inplace=True)
```

- Fills missing `Age` values with the **mode** (most frequent value).

---

### 3. **Age Cleanup and Conversion**

```python
import numpy as np
df['Age'] = df['Age'].astype(int)
```

- Detects and casts `Age` values to integers.
- This step fixes any fractional age values like `0.42`.

---

### 4. **Cabin Analysis and Feature Drop**

```python
df['Cabin'].unique()
df['Cabin'].nunique()
df_new = df.drop(columns = ['Name', 'PassengerId', 'Cabin', 'Ticket', 'Fare'])
```

- Analyzes the `Cabin` column (many missing values).
- Drops irrelevant or sparse columns that don’t help model performance.

---

### 5. **Encoding Categorical Features**

```python
df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})
```

- Converts `Sex` into binary format: male = 0, female = 1.

---

### 6. **Visualizing Outliers**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df_new['Age'])
plt.title("Box Plot For Age")
plt.show()
```

- Uses a **boxplot** to visualize outliers in the `Age` column.

---

### 7. **Removing Outliers (IQR Method)**

```python
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_new = remove_outliers_iqr(df_new, 'Age')
```

- Defines and applies a function to remove outliers using the **Interquartile Range (IQR)** method.

---

### 8. **Encoding 'Embarked' Column**

```python
embarked_dummies = pd.get_dummies(df_new['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
df_new = pd.concat([df_new, embarked_dummies], axis=1)
df_new.drop('Embarked', axis=1, inplace=True)
```

- Converts `Embarked` into dummy variables.
- Drops one dummy to avoid multicollinearity (using `drop_first=True`).

---

## ✅ Final Dataset

After these steps, the final `df_new` DataFrame is:
- Cleaned (no missing values)
- Numeric (all features converted for modeling)
- Outliers handled
- Ready for modeling (e.g., logistic regression, decision trees, etc.)

---

## 📊 Libraries Used

- **pandas** for data manipulation
- **numpy** for numerical operations
- **seaborn** and **matplotlib** for visualization

---

