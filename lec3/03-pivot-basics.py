
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # An Overview of Pandas Pivot Tables

# In[2]:


import numpy as np
import pandas as pd


# This exercise is inspired by Wes McKinney's [Python for Data Analysis](http://proquest.safaribooksonline.com.libproxy.berkeley.edu/book/programming/python/9781491957653)

# In[3]:


df = pd.read_csv("elections.csv")
df.head()


# ## Pivot Tables

# Recall from before that we were able to group the % Series by the "Party" and "Result" Series, allowing us to understand the average vote earned by each party under each election result.

# In[4]:


percent_grouped_by_party_and_result = df['%'].groupby([df['Party'], df['Result']])
percent_grouped_by_party_and_result.mean()


# Because we called `groupby` on a Series, the result of our aggregation operation was also a Series. However, I believe this data is more naturally expressed in a tabular format, with Party as the rows, and Result as the columns. The `pivot_table` operation is the natural way to achieve this data format.

# In[5]:


df_pivot = df.pivot_table(
    index='Party', # the rows (turned into index)
    columns='Result', # the column values
    values='%', # the field(s) to processed in each group
    aggfunc=np.mean, # group operation
)
df_pivot.head()


# The basic idea is that you specify a Series to be the `index` (i.e. rows) and a Series to be the `columns`. The data in the specified `values` is then grouped by all possible combinations of values that occur in the `index` and `columns` Series. These groups are then aggregated using the `aggfunc`, and arranged into a table that matches the requested `index` and `columns`. The diagram below summarizes how pivot tables are formed. (Diagram inspired by Joey Gonzales). Diagram source at [this link](https://docs.google.com/presentation/d/1FrYg6yd6B-CIgfWLWm4W8vBhfmJ6Qt9dKkN-mlN5AKU/edit#slide=id.g4131093782_0_89).
# 
# ![groupby](pivot_table_overview.png)

# For more on pivot tables, see [this excellent tutorial](http://pbpython.com/pandas-pivot-table-explained.html) by Chris Moffitt.

# ## List Arguments to pivot_table (Extra)

# The arguments to our pivot_table method can also be lists. A few examples are given below.

# If we pivot such that only our `columns` argument is a list, we end up with columns that are MultiIndexed.

# In[6]:


df_pivot = df.pivot_table(
    index='Party', # the rows (turned into index)
    columns=['Result', 'Candidate'], # the column values
    values='%', # the field(s) to processed in each group
    aggfunc=np.mean, # group operation
)
df_pivot.head()


# If we pivot such that only our `index` argument is a list, we end up with rows that are MultiIndexed.

# In[7]:


df_pivot = df.pivot_table(
    index=['Party', 'Candidate'], # the rows (turned into index)
    columns='Result',# the column values
    values='%', # the field(s) to processed in each group
    aggfunc=np.mean, # group operation
)
df_pivot


# If we pivot such that only our values argument is a list, then we again get a DataFrame with multi-indexed Columns.

# In[8]:


df_pivot = df.pivot_table(
    index='Party', # the rows (turned into index)
    columns='Result',# the column values
    values=['%', 'Year'], # the field(s) to processed in each group
    aggfunc=np.mean, # group operation
)
df_pivot


# Feel free to experiment with other possibilities!

# ## Submission
# 
# You're done!
# 
# Before submitting this assignment, ensure to:
# 
# 1. Restart the Kernel (in the menubar, select Kernel->Restart & Run All)
# 2. Validate the notebook by clicking the "Validate" button
# 
# Finally, make sure to **submit** the assignment via the Assignments tab in Datahub
