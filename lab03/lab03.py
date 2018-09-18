
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # Lab 3: Data Cleaning and Visualization
# 
# In this lab, you will be working with a dataset from the City of Berkeley containing data on calls to the Berkeley Police Department. Information about the dataset can be found [at this link](https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5).
# 
# **This assignment should be completed and submitted before Monday September 10, 2018 at 11:59 PM.**
# 
# ### Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about the labs, we ask that you **write your solutions individually**. If you do discuss the assignments with others, please **include their names** at the top of this notebook.

# ## Setup
# 
# Note that after activating matplotlib to display figures inline via the IPython magic `%matplotlib inline`, we configure a custom default figure size. Virtually every default aspect of matplotlib [can be customized](https://matplotlib.org/users/customizing.html).

# In[1]:


import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


plt.rcParams['figure.figsize'] = (12, 9)


# # Part 1: Cleaning and Exploring the Data
# 
# To retrieve the dataset, we will reuse the `ds100_utils.fetch_and_cache` utility. (This is the same function found in `ds100_utils.py` in `hw1`.)

# In[3]:


import ds100_utils

data_dir = 'data'
data_url = 'http://www.ds100.org/fa18/assets/datasets/lab03_data_fa18.zip'
file_name = 'lab03_data_fa18.zip'

dest_path = ds100_utils.fetch_and_cache(data_url=data_url, file=file_name, data_dir=data_dir)
print(f'Located at {dest_path}')


# We will now directly unzip the ZIP archive and start working with the uncompressed files.
# 
# Note: There is no single right answer to whether to work with compressed files in their compressed state or whether to uncompress them on disk permanently. If you e.g. need to work with multiple tools on the same files, or write many notebooks to analyze them, and they are not too large, it may be more convenient to uncompress them once.  But you may also have situations where you find it preferable to work with the compressed data directly.  
# 
# Python gives you tools for both approaches, and you should know how to perform both tasks in order to choose the one that best suits the problem at hand.
# 
# ---
# 
# Run the cell below to extract the zip file into the data directory.

# In[4]:


my_zip = zipfile.ZipFile(dest_path, 'r')
my_zip.extractall(data_dir)


# Now, we'll use a method of the `Pathlib.Path` class called `glob` to list all files in the `data` directory. You will find useful information in pathlib [docs](https://docs.python.org/3/library/pathlib.html).
# 
# Below, we use pathlib's `glob` method to store the list of all files' names from the `data_dir` directory in the variable `file_names`. These names should be strings that contain only the file name (e.g. `dummy.txt` not `data/dummy.txt`). The asterisk (*) character is used with the `glob` method to match any string.

# In[5]:


from pathlib import Path
data_dir_path = Path('data') # creates a Path object that points to the data directory
file_names = [x.name for x in data_dir_path.glob('*') if x.is_file()]
file_names


# Let's now load the CSV file we have into a `pandas.DataFrame` object.

# In[6]:


calls = pd.read_csv("data/Berkeley_PD_-_Calls_for_Service.csv")
calls.head()


# We see that the fields include a case number, the offense type, the date and time of the offense, the "CVLEGEND" which appears to be related to the offense type, a "CVDOW" which has no apparent meaning, a date added to the database, and the location spread across four fields.
# 
# Let's also check some basic information about these files using the `DataFrame.describe` and `DataFrame.info` methods.

# In[8]:


calls.info()
calls.describe()


# Notice that the functions above reveal type information for the columns, as well as some basic statistics about the numerical columns found in the DataFrame. However, we still need more information about what each column represents. Let's explore the data further in Question 1.
# 
# Before we go over the fields to see their meanings, the cell below will verify that all the events happened in Berkeley by grouping on the `City` and `State` columns. You should see that all of our data falls into one group.

# In[9]:


calls.groupby(["City","State"]).count()


# ## Question 1
# Above, when we called `head`, it seemed like `OFFENSE` and `CVLEGEND` both contained information about the type of event reported. What is the difference in meaning between the two columns? One way to probe this is to look at the `value_counts` for each Series.

# In[10]:


calls['OFFENSE'].value_counts().head(10)


# In[7]:


calls['CVLEGEND'].value_counts().head(10)


# ### Question 1a
# 
# Above, it seems like `OFFENSE` is more specific than `CVLEGEND`, e.g. "LARCENY" vs. "THEFT FELONY (OVER $950)". For those of you who don't know the word "larceny", it's a legal term for theft of personal property.
# 
# To get a sense of how many subcategories there are for each `OFFENSE`, set `calls_by_cvlegend_and_offense` equal to a multi-indexed series where the data is first indexed on the `CVLEGEND` and then on the `OFFENSE`, and the data is equal to the number of offenses in the database that match the respective `CVLEGEND` and `OFFENSE`. For example, calls_by_cvlegend_and_offense["LARCENY", "THEFT FROM PERSON"] should return 24.

# In[8]:


### BEGIN Solution

calls_by_cvlegend_and_offense = calls[['CVLEGEND', 'OFFENSE']].groupby(['CVLEGEND', 'OFFENSE']).size()
calls_by_cvlegend_and_offense
# calls_by_cvlegend_and_offense["LARCENY", "THEFT FROM PERSON"]

### END Solution

# YOUR CODE HERE
# raise NotImplementedError()


# In[13]:


assert calls_by_cvlegend_and_offense["LARCENY", "THEFT FROM PERSON"] == 24


# ### Question 1b
# 
# In the cell below, set `answer1b` equal to a list of strings corresponding to the possible values for `OFFENSE` when `CVLEGEND` is "LARCENY". You can type the answer manually, or you can create an expression that automatically extracts the names.

# In[9]:


# You may use this cell for your scratch work as long as you enter
# in your final answers in the answer1 variable.

### BEGIN Solution
answer1b = calls_by_cvlegend_and_offense["LARCENY"].index.values.tolist()
answer1b 
### END Solution

# YOUR CODE HERE
# raise NotImplementedError()


# In[10]:


assert isinstance(answer1b, list)
assert all([isinstance(elt, str) for elt in answer1b])
assert len(answer1b) == 3
# This makes sure all the values you gave are indeed values of calls['OFFENSE'] (no typos)
assert all([elt in calls['OFFENSE'].values for elt in answer1b])

# Helper for nbgrader tests
def ascii_sum(ans):
    return sum([sum(map(ord, s.strip())) for s in ans])

assert set([a.strip().upper() for a in answer1b]) == set(['THEFT FELONY (OVER $950)', 'THEFT FROM PERSON', 'THEFT MISD. (UNDER $950)'])


# ## Question 2
# 
# What are the five crime types of CVLEGEND that have the most crime events? You may need to use `value_counts` to find the answer.
# Save your results into `answer2` as a list of strings.
# 
# **Hint:** *The `keys` method of the Series class might be useful.*

# In[22]:


### BEGIN Solution
answer2_sort = calls.groupby('CVLEGEND').size().sort_values(ascending=False)[:5].index.values.tolist()
# answer2_sort = [pair[0] for pair in answer2_sort]
answer2 = answer2_sort
print(answer2)

# calls_by_cvlegend_and_offense.groupby('CVLEGENG').size()
# calls.groupby('CVLEGEND').size().sort_values(ascending=False)
# .sort_values(ascending=False).keys
### END Solution
# YOUR CODE HERE
# raise NotImplementedError()


# In[23]:


assert isinstance(answer2, list)
assert all([isinstance(elt, str) for elt in answer2])
# This makes sure all the values you gave are indeed values of the DataFrame (no typo)
assert all([elt in calls['CVLEGEND'].values for elt in answer2])

assert set(answer2) == set(['LARCENY', 'BURGLARY - VEHICLE', 'VANDALISM', 'DISORDERLY CONDUCT', 'ASSAULT'])


# ---
# # Part 2: Visualizing the Data
# 
# ## Pandas vs. Seaborn Plotting
# 
# Pandas offers basic functionality for plotting. For example, the `DataFrame` and `Series` classes both have a `plot` method. However, the basic plots generated by pandas are not particularly pretty. While it's possible to manually use matplotlib commands to make pandas plots look better, we'll instead use a high level plotting library called Seaborn that will take care of most of this for us.
# 
# As you learn to do data visualization, you may find the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html) and [Seaborn documentation](https://seaborn.pydata.org/api.html) helpful!

# As an example of the built-in plotting functionality of pandas, the following example uses `plot` method of the `Series` class to generate a `barh` plot type to visually display the value counts for `CVLEGEND`.

# In[ ]:


ax = calls['CVLEGEND'].value_counts().plot(kind='barh')
ax.set_ylabel("Crime Category")
ax.set_xlabel("Number of Calls")
ax.set_title("Number of Calls By Crime Type");


# By contrast, the Seaborn library provides a specific function `countplot` built for plotting counts. It operates directly on the DataFrame itself i.e. there's no need to call `value_counts()` at all. This higher level approach makes it easier to work with. Run the cell below, and you'll see that the plot is much prettier (albeit in a weird order).

# In[ ]:


ax = sns.countplot(data=calls, y="CVLEGEND")
ax.set_ylabel("Crime Category")
ax.set_xlabel("Number of Calls")
ax.set_title("Number of Calls By Crime Type");


# If we want the same ordering that we had in the pandas plot, we can use the order parameter of the `countplot` method. It takes a list of strings corresponding to the axis to be ordered. By passing the index of the `value_counts`, we get the order we want.

# In[ ]:


ax = sns.countplot(data=calls, y="CVLEGEND", order=calls["CVLEGEND"].value_counts(ascending=True).index);
ax.set_ylabel("Crime Category")
ax.set_xlabel("Number of Calls")
ax.set_title("Number of Calls By Crime Type");


# Voilà! Now we have a pretty bar plot with the bars ordered by size. Though `seaborn` appears to provide a superior plot from a aesthetic point of view, the `pandas` plotting library is also good to understand. You'll get practice using both libraries in the following questions.
# 
# ## An Additional Note on Plotting in Jupyter Notebooks
# 
# You may have noticed that many of our code cells involving plotting end with a semicolon (;). This prevents any extra output from the last line of the cell that we may not want to see. Try adding this to your own code in the following questions!

# ## Question 3
# 
# Now it is your turn to make some plots using `pandas` and `seaborn`. Let's start by looking at the distribution of calls over days of the week.
# 
# The CVDOW field isn't named helpfully and it is hard to see the meaning from the data alone. According to the website linked at the top of this notebook, CVDOW is actually indicating the day that events happened. 0->Sunday, 1->Monday ... 6->Saturday. 
# 
# ### Question 3a
# 
# Add a new column `Day` into the `calls` dataframe that has the string weekday (eg. 'Sunday') for the corresponding value in CVDOW. For example, if the first 3 values of `CVDOW` are `[3, 6, 0]`, then the first 3 values of the `Day` column should be `["Wednesday", "Saturday", "Sunday"]`.
# 
# **Hint:** *Try using the [Series.map](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) function on `calls["CVDOW"]`.  Can you assign this to the new column `calls["Day"]`?*

# In[ ]:


days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
day_indices = range(7)
indices_to_days_dict = dict(zip(day_indices, days)) # Should look like {0:"Sunday", 1:"Monday", ..., 6:"Saturday"}

### BEGIN Solution
calls['Day'] = calls['CVDOW'].map(lambda x: indices_to_days_dict[x])
calls
### END Solution

# YOUR CODE HERE
# raise NotImplementedError()


# In[ ]:


assert set(calls["Day"]) == {'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'}
assert list(calls["Day"][:5]) == ['Wednesday', 'Wednesday', 'Friday', 'Tuesday', 'Saturday']


# ### Question 3b
# 
# Run the cell below to create a `seaborn` plot. This plot shows the number of calls for each day of the week. Notice the use of the `rotation` argument in `ax.set_xticklabels`, which rotates the labels by 90 degrees.

# In[ ]:


ax = sns.countplot(data=calls, x='Day', order=days)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("Number of Calls")
ax.set_title("Number of Calls For Each Day of the Week");


# Now, let's make the same plot using `pandas`. Construct a vertical bar plot with the count of the number of calls (entries in the table) for each day of the week **ordered by the day of the week** (eg. `Sunday`, `Monday`, ...). Do not use `sns` for this plot. Be sure that your axes are labeled and that your plot is titled.
# 
# **Hint:** *Given a series `s`, and an array `coolIndex` that has the same entries as in `s.index`, `s[coolIndex]` will return a copy of the series in the same order as `coolIndex`.*

# In[ ]:


# YOUR CODE HERE
# raise NotImplementedError()

### BEGIN Solution
ax_3b = sns.countplot(data=calls, x='Day', order = calls['Day'].value_counts()[days].index)
ax_3b.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax_3b.set_ylabel("Number of Calls")
ax_3b.set_title("Number of Calls For Each Day of the Week");

### END Solution

# Leave this for grading purposes
ax_3b = plt.gca()


# In[ ]:


import matplotlib 
# Check ordering
assert np.alltrue(np.array([l.get_text() for l in ax_3b.xaxis.get_ticklabels()]) == days)
bars = [rect.get_height() for rect in ax_3b.get_children() 
        if isinstance(rect, matplotlib.patches.Rectangle) and rect.get_x() != 0.0
       ]
# Check values
assert np.alltrue(np.array(bars) == calls['Day'].value_counts()[days].values)


# ## Question 4
# 
# It seems weekdays generally have slightly more calls than Saturday or Sunday, but the difference does not look significant.  
# 
# We can break down into some particular types of events to see their distribution. For example, let's make a bar plot for the CVLEGEND "NOISE VIOLATION". Which day is the peak for "NOISE VIOLATION"?
# 
# ### Question 4a
# 
# This time, use `seaborn` to create a vertical bar plot of the number of total noise violations reported on each day of the week, again ordered by the days of the week starting with Sunday. Do not use `pandas` to plot.
# 
# **Hint:** *If you're stuck, use the code for the seaborn plot in Question 3b as a starting point.*

# In[ ]:


# YOUR CODE HERE
# raise NotImplementedError()

### BEGIN Solution
ax_4a = sns.countplot(data=calls[calls['CVLEGEND']=='NOISE VIOLATION'], x='Day', order = calls['Day'].value_counts()[days].index)
ax_4a.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax_4a.set_ylabel("Number of Calls Noise Violation")
ax_4a.set_title("Number of Calls Noise Violation For Each Day of the Week");
### END Solution

# Leave this for grading purposes
ax_4a = plt.gca()


# In[ ]:


import matplotlib 
# Check ordering
assert np.alltrue(np.array([l.get_text() for l in ax_4a.xaxis.get_ticklabels()]) == days)
bars = [rect.get_height() for rect in ax_4a.get_children() 
        if isinstance(rect, matplotlib.patches.Rectangle) and rect.get_x() != 0.0
       ]
# Check values
assert np.alltrue(np.array(bars)[4:] == [1, 1, 2])


# ### Question 4b
# 
# Do you realize anything interesting about the distribution of NOISE VIOLATION calls over a week? Type a 1-sentence answer in the cell below.

# The noise violation mainly happens on Saturday

# ## Question 5
# 
# Let's look at a similar distribution but for a crime we have much more calls data about. In the cell below, create the same plot as you did in Question 4, but now looking at instances of the CVLEGEND "FRAUD" (instead of "NOISE VIOLATION"). Use either `pandas` or `seaborn` plotting as you desire.

# In[ ]:


# YOUR CODE HERE
# raise NotImplementedError()


### BEGIN Solution
ax_5 = sns.countplot(data=calls[calls['CVLEGEND']=='FRAUD'], x='Day', order = calls['Day'].value_counts()[days].index)
ax_5.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax_5.set_ylabel("Number of Calls Fraud")
ax_5.set_title("Number of Calls Fraud For Each Day of the Week");
### END Solution

# Leave this for grading purposes
ax_5 = plt.gca()


# In[ ]:


import matplotlib 
# Check ordering
assert np.alltrue(np.array([l.get_text() for l in ax_5.xaxis.get_ticklabels()]) == days)
bars = [rect.get_height() for rect in ax_5.get_children() 
        if isinstance(rect, matplotlib.patches.Rectangle) and rect.get_x() != 0.0
       ]
# Check values
assert np.alltrue(np.array(bars) == np.array([27, 32, 29, 33, 35, 40, 12]))


# ## Question 6
# 
# ### Question 6a
# 
# Now let's look at the EVENTTM column which indicates the time for events. Since it contains hour and minute information, let's extract the hour info and create a new column named `Hour` in the `calls` dataframe. You should save the hour as an `int`. Then plot the frequency of each hour in the table (i.e., `value_counts()`) sorted by the hour of the day (i.e., `sort_index()`).
# 
# You will want to look into how to use:
# 
# * [Series.str.slice](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.slice.html#pandas.Series.str.slice) to select the substring.
# * [Series.astype](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.astype.html) to change the type.
# 
# **Hint:** *The `str` helper member of a series can be used to grab substrings.  For example, `calls["EVENTTM"].str.slice(3,5)` returns the minute of each hour of the `EVENTTM`.*

# In[24]:


# YOUR CODE HERE
# raise NotImplementedError()


### BEGIN Solution
calls['Hour'] = pd.to_numeric(calls['EVENTTM'].str.split(':').str[0])
# calls['Hour'].sort_index().value_counts()
ax_6a = sns.countplot(data=calls, x='Hour', order=calls['Hour'].value_counts().sort_index().index);
ax_6a.set_xlabel("Hour of the Day")
ax_6a.set_ylabel("Number of Calls")
ax_6a.set_title("Number of Calls Reporting Fraud For Each Hours of the Day");

### END Solution


# In[22]:


assert 'Hour' in calls.columns
assert set(calls["Hour"]) == set(range(24))
assert list(calls["Hour"][:5]) == [22, 21, 20, 8, 13]
assert np.allclose(calls['Hour'].mean(), 13.35823)


# The code in the cell below creates a pandas bar plot showing the number of FRAUD crimes committed at each hour of the day.

# In[23]:


ax = calls[calls["CVLEGEND"] == "FRAUD"]['Hour'].value_counts().sort_index().plot(kind='bar')
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Number of Calls")
ax.set_title("Number of Calls Reporting Fraud For Each Day of the Week");


# ### Question 6b
# 
# In the cell below, create a seaborn plot of the same data. Again, make sure you provide axes labels and a title for your plot.

# In[27]:


# YOUR CODE HERE
# raise NotImplementedError()

### BEGIN Solution
ax_6b = sns.countplot(data=calls[calls['CVLEGEND']=='FRAUD'], x='Hour', order=calls['Hour'].value_counts().sort_index().index)
ax_6b.set_xlabel("Hour of the Day")
ax_6b.set_ylabel("Number of Calls")
ax_6b.set_title("Number of Calls Reporting Fraud For Each Day of the Week");
### END Solution

# Leave this for grading purposes
ax_6b = plt.gca()


# In[28]:


import matplotlib 
bars = [rect.get_height() for rect in ax_6b.get_children() 
        if isinstance(rect, matplotlib.patches.Rectangle) and rect.get_x() != 0.0
       ]
true_counts = np.array([27, 1, 2, 2, 4, 2, 2, 4, 21, 9, 17, 10, 18, 15, 13, 12, 13, 6, 10, 8, 7, 1, 3, 1])
assert np.alltrue(np.array(bars) == true_counts)


# ### Question 6c
# 
# According to our plots, there seems to be a spike in calls reporting fraud at midnight. Do you trust that this spike is legitimate, or could there be an issue with our data? Explain your reasoning in 1-2 sentences in the cell below.

# I think this might be a issue of data. Since 210pm-2am the counts are generally small, it doesn't make sense that there is a huge spike at midnight. It might be because some data are labeleas "midnight" mistakenly.

# ## Question 7
# 
# In the cell below, we generate a boxplot which examines the hour of day of each crime broken down by the `CVLEGEND` value.  To construct this plot we used the [DataFrame.boxplot](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html) documentation.

# In[29]:


calls.boxplot(column="Hour", by='CVLEGEND', rot=90);


# While the pandas boxplot is informative, we can use seaborn to create a more visually-appealing plot. Using seaborn, regenerate a better box plot. See either the [ds 100 textbook](https://www.textbook.ds100.org/ch/06/viz_quantitative.html) or the [seaborn boxplot documentation](https://seaborn.pydata.org/generated/seaborn.boxplot.html).
# 
# Looking at your plot, which crime type appears to have the largest interquartile range? Put your results into `answer7` as a string.

# In[51]:


### BEGIN Solution
ax_7 = sns.boxplot(data=calls, y='Hour', x='CVLEGEND')
ax_7.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax_7.set_xlabel('CVLEGEND')
ax_7.set_ylabel('Hour')
ax_7.set_title('Boxplot groupby Hour');
# calls
### END Solution


# In[48]:


answer7 = "sex crime"

# Todo: Make a boxplot with seaborn
# YOUR CODE HERE
# raise NotImplementedError()


# In[49]:


assert isinstance(answer7, str)
assert answer7.upper() == "SEX CRIME"


# ## Congratulations
# 
# Congrats! You are finished with this assignment.

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
