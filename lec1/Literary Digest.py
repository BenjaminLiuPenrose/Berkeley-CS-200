
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[ ]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


tips = sns.load_dataset("tips")


# In[ ]:


tips


# In[ ]:


poll1936 = pd.read_csv("poll1936.csv")


# In[ ]:


poll1936


# In[ ]:


fig, ax = plt.subplots()
sns.set(font_scale=2.5)
fig.set_size_inches(11.7, 8.27)
sns.barplot(ax=ax, x="source", y="percentage", hue="who", data=poll1936)


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
