
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # Combined Lab + Discussion 2: Pandas Overview
# 
# **This assignment should be completed by Wednesday September 5, 2018 at 11:59 PM.**

# [Pandas](https://pandas.pydata.org/) is one of the most widely used Python libraries in data science. In this lab, you will learn commonly used data wrangling operations/tools in Pandas. We aim to give you familiarity with:
# 
# * Creating dataframes
# * Slicing data frames (ie. selecting rows and columns)
# * Filtering data (using boolean arrays)
# * Data Aggregation/Grouping dataframes
# * Merging dataframes
# 
# In this lab, you are going to use several pandas methods like `drop()`, `loc[]`, `groupby()`. You may press `shift+tab` on the method parameters to see the documentation for that method.

# **Just as a side note**: Pandas operations can be confusing at times and the documentation is not great, but it is OK to be stumped when figuring out why a piece of code is not doing what it's supposed to. We don't expect you to memorize all the different Pandas functions, just know the basic ones like `iloc[]`, `loc[]`, slicing, and other general dataframe operations. 
# 
# Throughout the semester, you will have to search through Pandas documentation and experiment, but remember it is part of the learning experience and will help shape you as a data scientist!

# ## Setup

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Creating DataFrames & Basic Manipulations
# 
# A [dataframe](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe) is a two-dimensional labeled data structure with columns of potentially different types.
# 
# The pandas [`DataFrame` function](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) provides at least two syntaxes to create a data frame.

# **Syntax 1: ** You can create a data frame by specifying the columns and values using a dictionary as shown below. 
# 
# The keys of the dictionary are the column names, and the values of the dictionary are lists containing the row entries.

# In[3]:


fruit_info = pd.DataFrame(
    data={'fruit': ['apple', 'orange', 'banana', 'raspberry'],
          'color': ['red', 'orange', 'yellow', 'pink']
          })
fruit_info


# **Syntax 2: ** You can also define a dataframe by specifying the rows like below. 
# 
# Each row corresponds to a distinct tuple, and the columns are specified separately.

# In[4]:


fruit_info2 = pd.DataFrame(
    [("red", "apple"), ("orange", "orange"), ("yellow", "banana"),
     ("pink", "raspberry")], 
    columns = ["color", "fruit"])
fruit_info2


# You can obtain the dimensions of a matrix by using the shape attribute dataframe.shape

# In[5]:


(num_rows, num_columns) = fruit_info.shape
num_rows, num_columns


# ### Question 1(a)
# 
# You can add a column by `dataframe['new column name'] = [data]`. Please add a column called `rank1` to the `fruit_info` table which contains a 1,2,3, or 4 based on your personal preference ordering for each fruit. 
# 

# In[6]:


# YOUR CODE HERE
fruit_info['rank1']=[1, 2, 3, 4]
# raise NotImplementedError()


# In[7]:


fruit_info


# In[12]:


assert fruit_info["rank1"].dtype == np.dtype('int64')
assert len(fruit_info["rank1"].dropna()) == 4


# ### Question 1(b)
# 
# You can ALSO add a column by `dataframe.loc[:, 'new column name'] = [data]`. This way to modify an existing dataframe is preferred over the assignment above. In other words, it is best that you use `loc[]`. Although using `loc[]` is more verbose, it is faster. (However, this tradeoff is more likely to be valuable in production than during interactive use.) We will explain in more detail what `loc[]` does, but essentially, the first parameter is for the rows and second is for columns. The `:` means keep all rows and the `new column name` indicates the column you are modifying or in this case adding. 
# 
# Please add a column called `rank2` to the `fruit_info` table which contains a 1,2,3, or 4 based on your personal preference ordering for each fruit. 
# 
# 

# In[8]:


# YOUR CODE HERE
fruit_info.loc[:, 'rank2']=[1, 2, 3, 4]
# raise NotImplementedError()


# In[9]:


fruit_info


# In[10]:


assert fruit_info["rank2"].dtype == np.dtype('int64')
assert len(fruit_info["rank2"].dropna()) == 4


# ### Question 2
# 
# Use the `.drop()` method to [drop](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html) the both the `rank1` and `rank2` columns you created. (Make sure to use the `axis` parameter correctly) 
# 
# Hint: Look through the documentation to see how you can drop multiple columns of a Pandas dataframe at once, it may involve a list.

# In[11]:


fruit_info_original = fruit_info.drop(['rank1', 'rank2'], axis=1)
# YOUR CODE HERE
# raise NotImplementedError()


# In[16]:


fruit_info_original


# In[17]:


assert fruit_info_original.shape[1] == 2


# ### Question 3
# 
# Use the `.rename()` method to [rename](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html) the columns of `fruit_info_original` so they begin with a capital letter. Set the `inplace` parameter correctly to change the `fruit_info_original` dataframe. (**Hint:** in Question 2, `drop` creates and returns a new dataframe instead of changing `fruit_info` because `inplace` by default is `False`)

# In[12]:


# YOUR CODE HERE
fruit_info_original.rename(columns={'color': 'Color', 'fruit': 'Fruit'},inplace=True)
# raise NotImplementedError()


# In[19]:


fruit_info_original


# In[13]:


assert fruit_info_original.columns[0] == 'Color'
assert fruit_info_original.columns[1] == 'Fruit'


# ### Babyname datasets
# Now that we have learned the basics, let's move on to the babynames dataset. Let's clean and wrangle the following data frames for the remainder of the lab.
# 
# First let's run the following cells to build the dataframe `baby_names`.
# The cells below download the data from the web and extract the data in a California region. There should be a total of 5933561  records.

# ### `fetch_and_cache` Helper
# 
# The following function downloads and caches data in the `data/` directory and returns the `Path` to the downloaded file

# In[14]:


def fetch_and_cache(data_url, file, data_dir="data", force=False):
    """
    Download and cache a url and return the file object.
    
    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded 
    
    return: The pathlib.Path object representing the file.
    """
    import requests
    from pathlib import Path
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir/Path(file)
    if force and file_path.exists():
        file_path.unlink()
    if force or not file_path.exists():
        print('Downloading...', end=' ')
        resp = requests.get(data_url)
        with file_path.open('wb') as f:
            f.write(resp.content)
        print('Done!')
    else:
        import time 
        birth_time = time.ctime(file_path.stat().st_ctime)
        print("Using cached version downloaded:", birth_time)
    return file_path


# Now, what in the world is the above `fetch_and_cache` function doing? Well, let's step through it and identify some of the key lines of the above function.
# 
# In Python, whenever you want to check if a file exists in a certain path, it is not sufficient to just have the string representation of the path, you need to create a Path object usign the `Path()` constructor. Essentially, after the Path object is created for the directory, a directory is created at that path location using the `mkdir()` method. Then, within the directory, a path for the file itself is created and if the path has already been linked (a.k.a file has already been created and put in the directory), then a new one is not created and instead uses the cached version.
# 
# The function `exists()` in the code above is one way to check if a file exists at a certain path when called on a path object. There is also another way this can be done using the `os` library in Python. If you decided to use the `os` library, you wouldn't need to create a Path object and rather pass in the the string representation of the path.
# 
# Now, going back to the code, if the path hasn't been linked, then the file is downloaded and created at the path location. 
# 
# The benefit of this function is that not only can you force when you want a new file to be downloaded using the `force` parameter, but in cases when you don't need the file to be re-downloaded, you can use the cached version and save download time.

# Below we use fetch and cache to download the `namesbystate.zip` zip file. 
# 
# **This might take a little while! Consider stretching.**

# In[15]:


data_url = 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'
namesbystate_path = fetch_and_cache(data_url, 'namesbystate.zip')


# The following cell builds the final full `baby_names` DataFrame. Here is documentation for [pd.concat](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.concat.html) if you want to know more about its functionality. 
# 

# In[20]:


import zipfile
zf = zipfile.ZipFile(namesbystate_path, 'r')

field_names = ['State', 'Sex', 'Year', 'Name', 'Count']

def load_dataframe_from_zip(zf, f):
    with zf.open(f) as fh: 
        return pd.read_csv(fh, header=None, names=field_names)

# List comprehension
states = [
    load_dataframe_from_zip(zf, f)
    for f in sorted(zf.filelist, key=lambda x:x.filename) 
    if f.filename.endswith('.TXT')
]

baby_names = pd.concat(states).reset_index(drop=True)


# In[17]:


import zipfile
import pandas as pd
zf = zipfile.ZipFile(namesbystate_path, 'r')

field_names = ['State', 'Sex', 'Year', 'Name', 'Count']

def load_dataframe_from_zip(zf, f):
    with zf.open(f) as fh: 
        return pd.read_csv(fh, header=None, names=field_names)

# List comprehension
states = [
    load_dataframe_from_zip(zf, f)
    for f in sorted(zf.filelist, key=lambda x:x.filename) 
    if f.filename.endswith('.TXT')
]
baby_names = pd.concat(states[:2])
queue = states[2:]
del states
while len(queue) != 0:
    if len(queue) % 2 == 1:
        baby_names = pd.concat([baby_names, queue[0]])
        queue = queue[1:]
    queue = [pd.concat(i) for i in zip(queue[::2],queue[1::2])]
baby_names = baby_names.reset_index(drop=True)


# In[21]:


baby_names.head()


# In[22]:


len(baby_names)


# ## Slicing Data Frames - selecting rows and columns
# 

# ### Selection Using Label
# 
# **Column Selection** 
# To select a column of a `DataFrame` by column label, the safest and fastest way is to use the `.loc` [method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html). General usage looks like `frame.loc[rowname,colname]`. (Reminder that the colon `:` means "everything").  For example, if we want the `color` column of the `ex` data frame, we would use : `ex.loc[:, 'color']`
# 
# - You can also slice across columns. For example, `baby_names.loc[:, 'Name':]` would give select the columns `Name` and the columns after.
# 
# - *Alternative:* While `.loc` is invaluable when writing production code, it may be a little too verbose for interactive use. One recommended alternative is the `[]` method, which takes on the form `frame['colname']`.
# 
# **Row Selection**
# Similarly, if we want to select a row by its label, we can use the same `.loc` method. In this case, the "label" of each row refers to the index (ie. primary key) of the dataframe.

# In[23]:


#Example:
baby_names.loc[2:5, 'Name']


# In[10]:


#Example:  Notice the difference between these two methods
baby_names.loc[2:5, ['Name']]


# The `.loc` actually uses the Pandas row index rather than row id/position of rows in the dataframe to perform the selection. Also, notice that if you write `2:5` with `loc[]`, contrary to normal Python slicing functionality, the end index is included, so you get the row with index 5. 
# 

# There is another Pandas slicing function called `iloc[]` which lets you slice the dataframe by row id and column id instead of by column name and row index (for `loc[]`). This is really the main difference between the 2 functions and it is important that you remember the difference and why you might want to use one over the other. 
# 
# In addition, with `iloc[]`, the end index is NOT included, like with normal Python slicing.
# 
# Here is an example of how we would get the 2nd, 3rd, and 4th rows with only the `Name` column of the `baby_names` dataframe using both `iloc[]` and `loc[]`. Observe the difference.

# In[24]:


baby_names.iloc[1:4, 3]


# In[25]:


baby_names.loc[1:3, "Name"]


# Lastly, we can change the index of a dataframe using the `set_index` method.

# In[26]:


#Example: We change the index from 0,1,2... to the Name column
df = baby_names[:5].set_index("Name") 
df


# We can now lookup rows by name directly:

# In[27]:


df.loc[['Mary', 'Anna'], :]


# However, if we still want to access rows by location we will need to use the integer loc (`iloc`) accessor:

# In[28]:


#Example: 
#df.loc[2:5,"Year"] You can't do this
df.iloc[1:4,2:3]


# ### Question 4
# 
# Selecting multiple columns is easy.  You just need to supply a list of column names.  Select the `Name` and `Year` **in that order** from the `baby_names` table.

# In[29]:


# YOUR CODE HERE
name_and_year = baby_names[['Name', 'Year']]
# raise NotImplementedError()


# In[17]:


name_and_year[:5]


# In[30]:


name_and_year.shape


# In[31]:


assert name_and_year.shape == (5933561, 2)
assert name_and_year.loc[0,"Name"] == "Mary"
assert name_and_year.loc[0,"Year"] == 1910


# As you may have noticed above, the .loc() method is a way to re-order the columns within a dataframe.

# ## Filtering Data

# ### Filtering with boolean arrays
# 
# Filtering is the process of removing unwanted material.  In your quest for cleaner data, you will undoubtedly filter your data at some point: whether it be for clearing up cases with missing values, culling out fishy outliers, or analyzing subgroups of your data set.  Note that compound expressions have to be grouped with parentheses. Example usage looks like `df[df[column name] < 5]]`.
# 
# For your reference, some commonly used comparison operators are given below.
# 
# Symbol | Usage      | Meaning 
# ------ | ---------- | -------------------------------------
# ==   | a == b   | Does a equal b?
# <=   | a <= b   | Is a less than or equal to b?
# >=   | a >= b   | Is a greater than or equal to b?
# <    | a < b    | Is a less than b?
# &#62;    | a &#62; b    | Is a greater than b?
# ~    | ~p       | Returns negation of p
# &#124; | p &#124; q | p OR q
# &    | p & q    | p AND q
# ^  | p ^ q | p XOR q (exclusive or)

# In the following we construct the DataFrame containing only names registered in California

# In[32]:


ca = baby_names[baby_names['State'] == "CA"]


# ### Question 5a
# Select the names in Year 2000 (for all baby_names) that have larger than 3000 counts. What do you notice?
# 
# (If you use `p & q` to filter the dataframe, make sure to use `df[df[(p) & (q)]]` or `df.loc[df[(p) & (q)]])`
# 
# **Remember** that both slicing and using `loc` will achieve the same result, it is just that `loc` is typically faster in production. You are free to use whichever one you would like.

# In[33]:


# YOUR CODE HERE
result=baby_names.loc[(baby_names['Year']==2000) & (baby_names['Count']>=3000)]
# raise NotImplementedError()


# In[38]:


result


# In[34]:


assert len(result) == 11
assert result["Count"].sum() == 38993
assert result["Count"].iloc[0] == 4339


# 
# ## Data Aggregration (Grouping Data Frames)
# 
# ### Question 6a
# To count the number of instances of each unique value in a `Series`, we can use the `value_counts()` [method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) as `df["col_name"].value_counts()`. Count the number of different names for each Year in `CA` (California).  (You may use the `ca` DataFrame created above.)
# 
# **Note:** *We are not computing the number of babies but instead the number of names (rows in the table) for each year.*

# In[35]:


num_of_names_per_year = ca['Year'].value_counts()
# YOUR CODE HERE
# raise NotImplementedError()


# In[36]:


num_of_names_per_year[:5]


# In[37]:


assert num_of_names_per_year[2007] == 7248
assert num_of_names_per_year[:5].sum() == 35607
assert num_of_names_per_year[1910] == 363
assert num_of_names_per_year[:15].sum() == 103699


# ### Question 6b
# Count the number of different names for each gender in `CA`. Does the result help explaining the findings in Question 5?

# In[38]:



num_of_names_per_gender = ca['Sex'].value_counts()
# YOUR CODE HERE
# raise NotImplementedError()


# In[39]:


num_of_names_per_gender


# In[40]:


assert num_of_names_per_gender["F"] > 200000
assert num_of_names_per_gender["F"] == 221084
assert num_of_names_per_gender["M"] == 153550


# ### Question 7: Groupby ###
# 
# Before we jump into using the `groupby` function in Pandas, let's recap how grouping works in general for tabular data through a guided set of questions based on a small toy dataset of movies and genres. 
# 
# **Note:** If you want to see a visual of how grouping of data works, here is a link to an animation from last week's slides: [Groupby Animation](http://www.ds100.org/sp18/assets/lectures/lec03/03-groupby_and_pivot.pdf)

# **Problem Setting:** This summer 2018, there were a lot of good and bad movies that came out. Below is a dataframe with 5 columns: name of the movie as a `string`, the genre of the movie as a `string`, the first name of the director of the movie as a `string`, the average rating out of 10 on Rotten Tomatoes as an `integer`, and the total gross revenue made by the movie as an `integer`. The point of these guided questions (parts a and b) below is to understand how grouping of data works in general, **not** how grouping works in code. We will worry about how grouping works in Pandas in 7c, which will follow.
# 
# Below is the `movies` dataframe we are using, imported from the `movies.csv` file located in the `lab02` directory.

# In[41]:


movies = pd.read_csv("movies.csv")
movies


# ### Question 7a
# 
# If we grouped the `movies` dataframe above by `genre`, how many groups would be in the output and what would be the groups? Assign `num_groups` to the number of groups created and fill in `genre_list` with the names of genres as strings that represent the groups.

# In[60]:


num_groups = len(movies['genre'].unique())
genre_list = []
# YOUR CODE HERE
genre_list = list(movies['genre'].unique())
# raise NotImplementedError()


# In[61]:


assert num_groups == 6
assert set(genre_list) == set(['Action & Adventure', 'Comedy', 'Science Fiction & Fantasy', 'Drama', 'Animation', 'Horror'])


# ### Question 7b
# 
# Whenever we group tabular data, it is usually the case that we need to aggregate values from the ungrouped column(s). If we were to group the `movies` dataframe above by `genre`, which column(s) in the `movies` dataframe would it make sense to aggregate if we were interested in finding how well each genre did in the eyes of people? Fill in `agg_cols` with the column name(s).

# In[62]:


agg_cols = ['rating', 'revenue']
# YOUR CODE HERE
movies.groupby('genre').mean()
# raise NotImplementedError()


# In[63]:


assert set(agg_cols) == set(['rating', 'revenue'])


# Now, let's see `groupby` in action, instead of keeping everything abstract. To aggregate data in Pandas, we use the `.groupby()` [function](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html). The code below will group the `movies` dataframe by `genre` and find the average revenue and rating for each genre. You can verify you had the same number of groups as what you answered in 7a. 

# In[64]:


movies.loc[:, ['genre', 'rating', 'revenue']].groupby('genre').mean()


# ### Question 7c
# 
# Let's move back to baby names and specifically, the `ca` dataframe. Find the sum of `Count` for each `Name` in the `ca` table. You should use `df.groupby("col_name").sum()`. Your result should be a Pandas Series.
# 
# **Note:** *In this question we are now computing the number of registered babies with a given name.*

# In[106]:


count_for_names = ca[['Name', 'Count']]
# YOUR CODE HERE
count_for_names = count_for_names.groupby('Name').sum().iloc[:,0]
# raise NotImplementedError()


# In[91]:


count_for_names.sort_values(ascending=False)[:5]


# In[66]:


assert count_for_names["Michael"] == 429827
assert count_for_names[:100].sum() == 95519
assert count_for_names["David"] == 371646
assert count_for_names[:1000].sum() == 1320144


# ### Question 7d
# 
# Find the sum of `Count` for each female name after year 1999 (`>1999`) in California.
# 

# In[67]:



female_name_count = ca[(ca['Year']>1999) & (ca['Sex']=='F')][['Name', 'Count']].groupby('Name').sum().iloc[:, 0]
# female_name_count = ca[(ca['Year']>1999) & (ca['Sex']=='F')][['Count']].groupby('Name').sum()
# YOUR CODE HERE
# raise NotImplementedError()


# In[68]:


female_name_count.sort_values(ascending=False)[:5]


# In[69]:


assert female_name_count["Emily"] == 48093
assert female_name_count[:100].sum() == 48596
assert female_name_count["Isabella"] == 45232
assert female_name_count[:10000].sum() == 3914766


# ### Question 8: Grouping Multiple Columns
# 
# Let's move back to the `movies` dataframe. Which of the following lines of code will output the following dataframe? Write your answer as either 1, 2, 3, or 4. Recall that the arguments to `pd.pivot_table` are as follows: `data` is the input dataframe, `index` includes the values we use as rows, `columns` are the columns of the pivot table, `values` are the values in the pivot table, and `aggfunc` is the aggregation function that we use to aggregate `values`.

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th>rating</th>
#       <th>5</th>
#       <th>6</th>
#       <th>7</th>
#       <th>8</th>
#     </tr>
#     <tr>
#       <th>genre</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>Action &amp; Adventure</th>
#       <td>208681866.0</td>
#       <td>129228350.0</td>
#       <td>318344544.0</td>
#       <td>6708147.0</td>
#     </tr>
#     <tr>
#       <th>Animation</th>
#       <td>374408165.0</td>
#       <td>NaN</td>
#       <td>NaN</td>
#       <td>NaN</td>
#     </tr>
#     <tr>
#       <th>Comedy</th>
#       <td>55383976.0</td>
#       <td>30561590.0</td>
#       <td>NaN</td>
#       <td>111705055.0</td>
#     </tr>
#     <tr>
#       <th>Drama</th>
#       <td>NaN</td>
#       <td>17146165.5</td>
#       <td>NaN</td>
#       <td>NaN</td>
#     </tr>
#     <tr>
#       <th>Horror</th>
#       <td>NaN</td>
#       <td>NaN</td>
#       <td>68765655.0</td>
#       <td>NaN</td>
#     </tr>
#     <tr>
#       <th>Science Fiction &amp; Fantasy</th>
#       <td>NaN</td>
#       <td>312674899.0</td>
#       <td>NaN</td>
#       <td>NaN</td>
#     </tr>
#   </tbody>
# </table>

# 1) `pd.pivot_table(data=movies, index='genre', columns='rating', values='revenue', aggfunc=np.mean)`
# 
# 2) `movies.groupby(['genre', 'rating'])['revenue'].mean()`
# 
# 3) `pd.pivot_table(data=movies, index='rating', columns='genre', values='revenue', aggfunc=np.mean)`
# 
# 4) `movies.groupby('revenue')[['genre', 'rating']].mean()`

# In[70]:


q7e_answer = 1
# YOUR CODE HERE
# raise NotImplementedError()


# In[71]:


assert q7e_answer == 1


# ### Question 9: Merging
# 

# #### Question 9(a)
# 
# Time to put everything together! Merge `movies` and `count_for_names` to find the number of registered baby names for each director. Only include names that appear in both `movies` and `count_for_names`.
# 
# **Hint:** You might need to convert the `count_for_names` series to a dataframe. Take a look at the ``to_frame`` method of a series to do this. 

# Your first row should look something like this:
# 
# **Note**: It is ok if you have 2 separate columns with names instead of just one column.
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>director</th>
#       <th>genre</th>
#       <th>movie</th>
#       <th>rating</th>
#       <th>revenue</th>
#       <th>Count</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>David</td>
#       <td>Action &amp; Adventure</td>
#       <td>Deadpool 2</td>
#       <td>7</td>
#       <td>318344544</td>
#       <td>371646</td>
#     </tr>
#   </tbody>
# </table>
# </table>

# In[108]:


# count_for_names['director']=count_for_names.to_frame()
# count_for_names
# count_for_names = count_for_names.to_frame()
# count_for_names['director']=count_for_names.index
# count_for_names
# count_for_names.rename(columns={'Name': 'director'}, inplace=True)
# count_for_names.to_frame()
merged_df = pd.merge(movies, count_for_names.to_frame(), left_on='director', right_index=True, how='inner').loc[:, ['director', 'genre', 'movie', 'rating', 'revenue', 'Count']]
# YOUR CODE HERE
# raise NotImplementedError()


# In[109]:


assert merged_df.loc[0, 'Count'] == 371646
assert merged_df.loc[7, 'Count'] == 7236
assert merged_df.loc[12, 'Count'] == 6586
assert merged_df['Count'].sum() == 861694
assert len(merged_df) == 14


# #### Question 9(b)
# 
# How many directors in the original `movies` table did not get included in the `merged_df` dataframe? Please explain your answer in 1-2 sentences.

# In[110]:


q_9b = movies.shape[0] - merged_df.shape[0]
# YOUR CODE HERE
# raise NotImplementedError()


# 4 directors' name is not matched in the baby_names database

# In[111]:


assert q_9b == 4


# #### You are done! Remember to validate and submit via JupyterHub

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
