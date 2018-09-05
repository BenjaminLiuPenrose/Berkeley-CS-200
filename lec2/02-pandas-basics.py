
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')
sns.set_context("notebook")


# ## Reading in DataFrames from Files

# Pandas has a number of very useful file reading tools. You can see them enumerated by typing "pd.re" and pressing tab. We'll be using read_csv today. 

# In[3]:


elections = pd.read_csv("elections.csv")
elections # if we end a cell with an expression or variable name, the result will print


# We can use the head command to show only a few rows of a dataframe.

# In[4]:


elections.head(7)


# There is also a tail command.

# In[5]:


elections.tail(7)


# The read_csv command lets us specify a column to use an index. For example, we could have used Year as the index.

# In[6]:


elections_year_index = pd.read_csv("elections.csv", index_col = "Year")
elections_year_index.head(5)


# Alternately, we could have used the set_index commmand.

# In[7]:


elections_party_index = elections.set_index("Party")
elections_party_index.head(5)


# The set_index command (along with all other data frame methods) does not modify the dataframe. That is, the original "elections" is untouched. Note: There is a flag called "inplace" which does modify the calling dataframe.

# In[8]:


elections.head() #the index remains unchanged


# By contast, column names MUST be unique. For example, if we try to read in a file for which column names are not unique, Pandas will automatically any duplicates.

# In[9]:


dups = pd.read_csv("duplicate_columns.csv")
dups


# ## The [] Operator

# The DataFrame class has an indexing operator [] that lets you do a variety of different things. If your provide a String to the [] operator, you get back a Series corresponding to the requested label.

# In[10]:


elections["Candidate"].head(6)


# The [] operator also accepts a list of strings. In this case, you get back a DataFrame corresponding to the requested strings.

# In[11]:


elections[["Candidate", "Party"]].head(6)


# A list of one label also returns a DataFrame. This can be handy if you want your results as a DataFrame, not a series.

# In[12]:


elections[["Candidate"]].head(6)


# Note that we can also use the to_frame method to turn a Series into a DataFrame.

# In[13]:


elections["Candidate"].to_frame()


# The [] operator also accepts numerical slices as arguments. In this case, we are indexing by row, not column!

# In[14]:


elections[0:3]


# If you provide a single argument to the [] operator, it tries to use it as a name. This is true even if the argument passed to [] is an integer. 

# In[ ]:


#elections[0] #this does not work, try uncommenting this to see it fail in action, woo


# The following cells allow you to test your understanding.

# In[15]:


weird = pd.DataFrame({1:["topdog","botdog"], "1":["topcat","botcat"]})
weird


# In[16]:


weird[1] #try to predict the output


# In[17]:


weird[["1"]] #try to predict the output


# In[18]:


weird[1:] #try to predict the output


# ## Boolean Array Selection

# The `[]` operator also supports array of booleans as an input. In this case, the array must be exactly as long as the number of rows. The result is a filtered version of the data frame, where only rows corresponding to True appear.

# In[19]:


elections[[False, False, False, False, False, 
          False, False, True, False, False,
          True, False, False, False, True,
          False, False, False, False, False,
          False, False, True]]


# One very common task in Data Science is filtering. Boolean Array Selection is one way to achieve this in Pandas. We start by observing logical operators like the equality operator can be applied to Pandas Series data to generate a Boolean Array. For example, we can compare the 'Result' column to the String 'win':

# In[20]:


elections.head(5)


# In[21]:


iswin = elections['Result'] == 'win'
iswin.head(5)


# In[22]:


elections[iswin]


# The output of the logical operator applied to the Series is another Series with the same name and index, but of datatype boolean. The entry with index i represents the result of the application of that operator to the entry of the original Series with index i.

# In[23]:


elections[elections['Party'] == 'Independent']


# In[24]:


elections['Result'].head(5)


# These boolean Series can be used as an argument to the [] operator. For example, the following code creates a DataFrame of all election winners since 1980.

# In[25]:


elections.loc[iswin]


# Above, we've assigned the result of the logical operator to a new variable called `iswin`. This is uncommon. Usually, the series is created and used on the same line. Such code is a little tricky to read at first, but you'll get used to it quickly.

# In[26]:


elections[elections['Result'] == 'win']


# We can select multiple criteria by creating multiple boolean Series and combining them using the `&` operator.

# In[27]:


elections[(elections['Result'] == 'win')
          & (elections['%'] < 50)]

# __and__ overrides & not and.


# ## Loc and ILOC

# In[28]:


elections.head(5)


# In[29]:


elections.loc[[0, 1, 2, 3, 4], ['Candidate','Party', 'Year']]


# Loc also supports slicing (for all types, including numeric and string labels!). Note that the slicing for loc is **inclusive**, even for numeric slices.

# In[30]:


elections.loc[0:4, 'Candidate':'Year']


# If we provide only a single label for the column argument, we get back a Series.

# In[32]:


elections.loc[0:4, 'Candidate']


# If we want a data frame instead and don't want to use to_frame, we can provde a list containing the column name.

# In[31]:


elections.loc[0:4, ['Candidate']]


# If we give only one row but many column labels, we'll get back a Series corresponding to a row of the table. This new Series has a neat index, where each entry is the name of the column that the data came from.

# In[33]:


elections.loc[0, 'Candidate':'Year']


# In[34]:


elections.loc[[0], 'Candidate':'Year']


# If we omit the column argument altogether, the default behavior is to retrieve all columns. 

# In[35]:


elections.loc[[2, 4, 5]]


# Loc also supports boolean array inputs instead of labels. If the arrays are too short, loc assumes the missing values are False.

# In[36]:


elections.loc[[True, False, False, True], [True, False, False, True]]


# In[37]:


elections.loc[[0, 3], ['Candidate', 'Year']]


# We can use boolean array arguments for one axis of the data, and labels for the other.

# In[38]:


elections.loc[[True, False, False, True], 'Candidate':'%']


# Boolean Series are also boolean arrays, so we can use the Boolean Array Selection from earlier using loc as well.

# In[39]:


elections.loc[(elections['Result'] == 'win') & (elections['%'] < 50), 
              'Candidate':'%']


# Let's do a quick example using data with string-labeled rows instead of integer labeled rows, just to make sure we're really understanding loc.

# In[40]:


mottos = pd.read_csv("mottos.csv", index_col = "State")
mottos.head(5)


# As you'd expect, the rows to extract can be specified using slice notation, even if the rows have string labels instead of integer labels.

# In[41]:


mottos.loc['California':'Florida', ['Motto', 'Language']]


# Sometimes students are so used to thinking of rows as numbered that they try the following, which will not work.

# In[44]:


mottos_extreme = pd.read_csv("mottos_extreme.csv", index_col='State')
mottos_extreme.loc['California']


# In[45]:


mottos_extreme.loc['California':'Delaware']
#did i mess up my experiment or is the answer?


# ### iloc

# loc's cousin iloc is very similar, but is used to access based on numerical position instead of label. For example, to access to the top 3 rows and top 3 columns of a table, we can use [0:3, 0:3]. iloc slicing is **exclusive**, just like standard Python slicing of numerical values.

# In[46]:


elections.head(5)


# In[47]:


elections.iloc[0:3, 0:3]


# In[48]:


mottos.iloc[0:3, 0:3]


# We will use both loc and iloc in the course. Loc is generally preferred for a number of reasons, for example: 
# 
# 1. It is harder to make mistakes since you have to literally write out what you want to get.
# 2. Code is easier to read, because the reader doesn't have to know e.g. what column #31 represents.
# 3. It is robust against permutations of the data, e.g. the social security administration switches the order of two columns.
# 
# However, iloc is sometimes more convenient. We'll provide examples of when iloc is the superior choice.

# ## Handy Properties and Utility Functions for Series and DataFrames

# The head, shape, size, and describe methods can be used to quickly get a good sense of the data we're working with. For example:

# In[49]:


mottos.head(5)


# In[50]:


mottos.size


# The fact that the size is 200 means our data file is relatively small, with only 200 total entries.

# In[51]:


mottos.shape


# Since we're looking at data for states, and we see the number 50, it looks like we've mostly likely got a complete dataset that omits Washington D.C. and U.S. territories like Guam and Puerto Rico.

# In[52]:


mottos.describe()


# Above, we see a quick summary of all the data. For example, the most common language for mottos is Latin, which covers 23 different states. Does anything else seem surprising?

# We can get a direct reference to the index using .index.

# In[53]:


mottos.index


# We can also access individual properties of the index, for example, `mottos.index.name`.

# In[54]:


mottos.index.name


# This reflects the fact that in our data frame, the index IS the state!

# In[55]:


mottos.head(2)


# It turns out the columns also have an Index. We can access this index by using `.columns`.

# In[56]:


mottos.columns


# There are also a ton of useful utility methods we can use with Data Frames and Series. For example, we can create a copy of a data frame sorted by a specific column using `sort_values`.

# In[57]:


elections.sort_values('%')


# As mentioned before, all Data Frame methods return a copy and do **not** modify the original data structure, unless you set inplace to True.

# In[58]:


elections.head(5)


# If we want to sort in reverse order, we can set `ascending=False`.

# In[60]:


elections.sort_values('%', ascending=False)


# We can also use `sort_values` on Series objects.

# In[61]:


mottos['Language'].sort_values().head(10)


# For Series, the `value_counts` method is often quite handy.

# In[62]:


elections['Party'].value_counts()


# In[63]:


mottos['Language'].value_counts()


# Also commonly used is the `unique` method, which returns all unique values as a numpy array.

# In[64]:


mottos['Language'].unique()


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
