
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv("enrollments.csv")
df.head()


# In[4]:


df = df[["Term", "Subject", "Number", "Title", "Enrollment Cnt", "Instructor"]]
df.head()


# ## Challenge One

# Try to find all Spring offerings of this course. Note, this dataset only contains Spring offerings, so there's no need to filter based on semester. The official "Number" for this class is "C100".

# In[5]:


df[df["Number"] == "C100"]


# ## Challenge Two

# Create a series where each row correspond to one subject (e.g. English), and each value corresponds to the average number of students for courses in that subject. For example, your series might have a row saying that the average number of students in a Computer Science class is 88.

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#  

# In[6]:


enrollment_grouped_by_subject = df["Enrollment Cnt"].groupby(df["Subject"])


# In[7]:


enrollment_grouped_by_subject.mean()


# ## Challenge Three

# Create a multi-indexed series where each row corresponds to one subject (e.g. English) offered during one semester (e.g. Spring 2017), and each value corresponds to the maximum number of students for courses in that subject during that semester. For example, you might have a row saying that the maximum number of students in a computer science course during Spring 2012 was 575.

# In[ ]:



























# In[8]:


enrollment_grouped_by_subject_and_term = df["Enrollment Cnt"].groupby([df["Subject"], df["Term"]])


# In[9]:


enrollment_grouped_by_subject_and_term.max()


# ## Challenge Four

# Try to compute the size of the largest class ever taught by each instructor. This challenge is stated more vaguely on purpose. You'll have to decide what the data structure looks like. Your result should be sorted in decreasing order of class size.

# In[ ]:





























# In[10]:


enrollment_grouped_by_instructor = df["Enrollment Cnt"].groupby(df["Instructor"])


# In[11]:


enrollment_grouped_by_instructor.max().sort_values(ascending=False)


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
