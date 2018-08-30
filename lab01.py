
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # Lab 1
# 
# Welcome to the first lab of Data 100! This lab is meant to help you familiarize yourself with JupyterHub and introduce you to `matplotlib`, a python visualization library. Here is the documentation: https://matplotlib.org/api/pyplot_api.html
# 
# ## Course Policies
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about the labs, we ask that you **write your solutions individually**. If you do discuss the assignments with others please **include their names** at the top of this notebook.
# 
# **This assignment should be completed and submitted before Monday August 27, 2018 at 11:59 PM.**
# 

# ### Running a Cell 
# 
# Try running the following cell.  If you are unfamiliar with Jupyter Notebooks, consider skimming [this tutorial](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb) or selecting **Help -> User Interface Tour** in the menu above. 

# In[2]:


print("Hello World!")


# Even if you are familiar with Jupyter, we strongly encourage you to become proficient with keyboard shortcuts (this will save you time in the future). To learn about keyboard shortcuts, go to **Help -> Keyboard Shortcuts** in the menu above. 
# 
# Here are a few we like:
# 1. `ctrl`+`return` : *Evaluate the current cell*
# 1. `shift`+`return`: *Evaluate the current cell and move to the next*
# 1. `esc` : *command mode* (may need to press before using any of the commands below)
# 1. `a` : *create a cell above*
# 1. `b` : *create a cell below*
# 1. `dd` : *delete a cell*
# 1. `m` : *convert a cell to markdown*
# 1. `y` : *convert a cell to code*

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# You might be wondering what `%matplotlib inline` does? `%matplotlib inline` is a magic command in Python that visualizes and stores any plots you draw or any plots we provide directly in the notebook.

# ---
# ## Part A
# We're going to start by going through the official `pyplot` tutorial. Please go through the [tutorial](pyplot.ipynb) and familiarize yourself with the basics of `pyplot`. This should take roughly 25 minutes.

# ### A note on `np.arange` and `np.linspace`
# 
# Note that the tutorial uses `np.arange`. While this is fine in some cases, we generally prefer to use `np.linspace`. `np.linspace(a, b, N)` divides the interval `[a, b]` into N points, while `np.arange(a, b, s)` will step from `a` to `b` with a fixed step size `s`.
# 
# One thing to keep in mind is that `np.linspace` always includes both end points while `np.arange` will *not* include the second end point `b`. For this reason, when we are plotting ranges of values we tend to prefer `np.linspace`.
# 
# The following two functions return the same result, but notice how their parameters differ.

# In[4]:


np.arange(-5, 6, 1.0)


# In[5]:


np.linspace(-5, 5, 11)


# ### Another tip
# 
# If you are ever confused about how a function works or behaves, click to the right of the function name, or inside the parentheses following the function. Then type `Shift` + `Tab`, and a window with the function's signature and docstring will appear. Holding down `Shift` and pressing `Tab` multiple times will bring up more and more detailed levels of documentation. Try this out in the cells above on `np.arange` and `np.linspace`!
# 

# ---
# ## Part B
# Now that you're familiar with the basics of `pyplot`, let's practice with some `pyplot` questions.

# ### Question 1a
# Let's visualize the function $f(t) = 3\sin(2\pi t)$. Set the x limit of all figures to $[0, \pi]$ and the y limit to $[-10, 10]$. Plot the sine function using `plt.plot` with 30 red plus signs. Additionally, make sure the x ticks are labeled $[0, \frac{\pi}{2}, \pi]$, and that your axes are labeled as well.
# 
# Your plot should look like the following:
# 
# ![1a.png](1a.png)
# 
# Hint 1: You can set axis bounds with `plt.axis`.
# 
# Hint 2: You can set xticks and labels with `plt.xticks`.
# 
# Hint 3: Make sure you add `plt.xlabel`, `plt.ylabel`, `plt.title`

# In[11]:


# YOUR CODE HERE
import math
x = np.linspace(0, math.pi, 50)
y = 3*np.sin(2*math.pi*x)
plt.plot(x, y, 'r+')
plt.axis([0, math.pi, -10, 10])
plt.xticks(np.linspace(0, math.pi, 3), ("0", "pi/2", "pi"))
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('f(t)=3sin(2*pi*t)')
# raise NotImplementedError()


# ### Question 1b
# Suppose we want to visualize the function $g(t) = a \cdot \sin(2 \pi f t)$ while varying the values $f, a$. Generate a 2 by 2 plot that plots the function $g(t)$ as a line plot with values $f = 2, 8$ and $a = 2, 8$. Since there are 2 values of $f$ and 2 values of $a$ there are a total of 4 combinations, hence a 2 by 2 plot. The rows should vary in $a$ and the columns should vary in $f$.
# 
# Set the x limit of all figures to $[0, \pi]$ and the y limit to $[-10, 10]$. The figure size should be 8 by 8. Make sure to label your x and y axes with the appropriate value of $f$ or $a$. Additionally, make sure the x ticks are labeled $[0, \frac{\pi}{2}, \pi]$. Your overall plot should look something like this:
# 
# ![2by2](1b.png)
# 
# Hint 1: Modularize your code and use loops.
# 
# Hint 2: Are your plots too close together such that the labels are overlapping with other plots? Look at the [`plt.subplots_adjust`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html) function.
# 
# Hint 3: Having trouble setting the x-axis ticks and ticklabels? Look at the [`plt.xticks`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html) function.
# 
# Hint 4: You can add title to overall plot with `plt.suptitle`.

# In[24]:


# YOUR CODE HERE
x = np.linspace(0, math.pi, 500)
a_f = [(2, 2), (8, 2), (2, 8), (8, 8)]
y ={}
num_row, num_col = 2, 2
plt.figure(10)
for i in range(num_row*num_col):
    plt.subplot(num_row, num_col, i+1)
    (a, f) = a_f[i]
    y[i] = a*np.sin(2*math.pi*f*x)
    plt.plot(x, y[i])
    plt.axis([0, math.pi, -10, 10])
    plt.xticks(np.linspace(0, math.pi, 3), ("0", "pi/2", "pi"))
    plt.xlabel('a: {}'.format(a))
    plt.ylabel('f: {}'.format(f))
plt.suptitle('Sine waves with varying a=[2,8], f=[2,8]')
plt.subplots_adjust(wspace=1, hspace=1)


# raise NotImplementedError()


# ### Question 2
# We should also familiarize ourselves with looking up documentation and learning how to read it. Below is a section of code that plots a basic wireframe. Replace each `Your answer here` with a description of what the line above does, what the arguments being passed in are, and how the arguments are used in the function. For example,
# 
# `np.arange(2, 5, 0.2)`
# 
# `# This returns an array of numbers from 2 to 5 with an interval size of 0.2`
# 
# Hint: The `Shift` + `Tab` tip from earlier in the notebook may help here.

# In[31]:


from mpl_toolkits.mplot3d import axes3d

u = np.linspace(1.5*np.pi, -1.5*np.pi, 100)
# This returns an array of numbers from 1.5pi to -1.5pi with 100 bins
[x,y] = np.meshgrid(u, u)
# This return coordinate matrices [x, y] from coordinate vectors (u, u), aka corordinate matrix [(x,y)] where x=y=u in this case
squared = np.sqrt(x.flatten()**2 + y.flatten()**2)
z = np.cos(squared)
# var squared is the point distance to the origin (aka l2 norm)
# z is the the cosine value of the squared
z = z.reshape(x.shape)
# This reshape z into x's shape (as well as the same as y's shape, 100*100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# add a subplot at position 1 of a (1,1) grid
ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
# plot wireframe of the x,y,z data
ax.view_init(elev=50., azim=30)
# view the plot
plt.savefig("figure1.png")
# YOUR CODE HERE
# raise NotImplementedError()


# ### Question 3
# Do you think that eating french fries with mayonnaise is a crime?  
# Tell us what you think in the following Markdown cell. :)

# ##### Answer:
# Emmm. I don't think so. Since it is his/her **freedom** to eat FF in his/her own way.

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
