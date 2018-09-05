
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[ ]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # DataTables, Indexes, Pandas, and Seaborn
# 
# ## Some useful (free) resources
# 
# Introductory:
# 
# * [Getting started with Python for research](https://github.com/TiesdeKok/LearnPythonforResearch), a gentle introduction to Python in data-intensive research.
# 
# * [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/index.html), by Jake VanderPlas, another quick Python intro (with notebooks).
# 
# Core Pandas/Data Science books:
# 
# * [The Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/), by Jake VanderPlas.
# 
# * [Python for Data Analysis, 2nd Edition](http://proquest.safaribooksonline.com/book/programming/python/9781491957653), by  Wes McKinney, creator of Pandas. [Companion Notebooks](https://github.com/wesm/pydata-book)
# 
# * [Effective Pandas](https://github.com/TomAugspurger/effective-pandas), a book by Tom Augspurger, core Pandas developer.
# 
# 
# Complementary resources:
# 
# * [An introduction to "Data Science"](https://github.com/stefanv/ds_intro), a collection of Notebooks by BIDS' [Stéfan Van der Walt](https://bids.berkeley.edu/people/st%C3%A9fan-van-der-walt).
# 
# * [Effective Computation in Physics](http://proquest.safaribooksonline.com/book/physics/9781491901564), by Kathryn D. Huff; Anthony Scopatz. [Notebooks to accompany the book](https://github.com/physics-codes/seminar). Don't be fooled by the title, it's a great book on modern computational practices with very little that's physics-specific.
# 
# 
# OK, let's load and configure some of our core libraries (as an aside, you can find a nice visual gallery of available matplotlib sytles [here](https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html)).

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')
sns.set_context("notebook")


# ## Getting the Data

# https://www.ssa.gov/OACT/babynames/index.html
# 
# https://www.ssa.gov/data
# 
# As we saw before, we can download data from the internet with Python, and do so only if needed:

# In[4]:


import requests
from pathlib import Path

namesbystate_path = Path('namesbystate.zip')
data_url = 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'

if not namesbystate_path.exists():
    print('Downloading...', end=' ')
    resp = requests.get(data_url)
    with namesbystate_path.open('wb') as f:
        f.write(resp.content)
    print('Done!')


# Let's use Python to understand how this data is laid out:

# In[5]:


import zipfile
zf = zipfile.ZipFile(namesbystate_path, 'r')
print([f.filename for f in zf.filelist])


# We can pull the PDF readme to view it, but let's operate with the rest of the data in its compressed state:

# In[6]:


zf.extract('StateReadMe.pdf')


# Let's have a look at the California data, it should give us an idea about the structure of the whole thing:

# In[7]:


ca_name = 'CA.TXT'
with zf.open(ca_name) as f:
    for i in range(10):
        print(f.readline().rstrip().decode())


# This is equivalent (on macOS or Linux) to extracting the full `CA.TXT` file to disk and then using the `head` command (if you're on Windows, don't try to run the cell below):

# In[8]:


zf.extract(ca_name)
get_ipython().system('head {ca_name}')


# In[9]:


get_ipython().system('cat /tmp/environment.yml')


# In[10]:


get_ipython().system('echo {ca_name}')


# A couple of practical comments:
# 
# * The above is using special tricks in IPython that let you call operating system commands via `!cmd`, and that expand Python variables in such commands with the `{var}` syntax. You can find more about IPython's special tricks [in this tutorial](https://github.com/ipython/ipython-in-depth/blob/master/examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb).
# 
# * `head` doesn't work on Windows, though there are equivalent Windows commands. But by using Python code, even if it's a little bit more verbose, we have a 100% portable solution.
# 
# * If the `CA.TXT` file was huge, it would be wasteful to write it all to disk only to look at the start of the file.
# 
# The last point is an important, and general theme of this course: we need to learn how to operate with data only on an as-needed basis, because there are many situations in the real world where we can't afford to brute-force 'download all the things'.
# 
# Let's remove the `CA.TXT` file to make sure we keep working with our compressed data, as if we couldn't extract it:

# In[11]:


import os; os.unlink(ca_name)


# ## Question 1: What was the most popular name in CA last year?

# In[12]:


import pandas as pd

field_names = ['State', 'Sex', 'Year', 'Name', 'Count']
with zf.open(ca_name) as fh:
    ca = pd.read_csv(fh, header=None, names=field_names)
ca.head()


# ### Indexing Review
# 
# Let's play around a bit with our indexing techniques from earlier today.

# In[13]:


ca['Count'].head()


# In[14]:


ca[0:3]


# In[ ]:


#ca[0]


# In[15]:


ca.iloc[:3, -2:]


# In[16]:


ca.loc[0:3, 'State']


# In[17]:


ca['Name'].head()


# In[18]:


ca[['Name']].head()


# In[19]:


ca[ca['Year'] == 2017].tail()


# ## Understanding the Data

# In[20]:


ca.head()


# We can get a sense for the shape of our data:

# In[21]:


ca.shape


# In[22]:


ca.size  # rows x columns


# Pandas will give us a summary overview of the *numerical* data in the DataFrame:

# In[23]:


ca.describe()


# And let's look at the *structure* of the DataFrame:

# In[24]:


ca.index


# ### Sorting

# What we've done so far is NOT exploratory data analysis. We were just playing around a bit with the capabilities of the pandas library. Now that we're done, let's turn to the problem at hand: Identifying the most common name in California last year.

# In[25]:


ca2017 = ca[ca['Year'] == 2017]
ca_sorted = ca2017.sort_values('Count', ascending=False).head(10)
ca_sorted


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
