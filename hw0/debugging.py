
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[ ]:


NAME = "Benjamin Liu"
COLLABORATORS = ""


# ---

# # Intro to Debugging with pdb
# This notebook will provide a brief introduction to debugging with `pdb`. It will walk through a few examples and explain different commands and how they interact with your code. This is not a complete tutorial, so please refer to the documentation: https://docs.python.org/3/library/pdb.html#module-pdb, https://github.com/spiside/pdb-tutorial and https://realpython.com/python-debugging-pdb/.
# 
# This notebook is based off of the second link, so please check it out for a more thorough introduction to `pdb`.
# 
# Here's a useful cheatsheet of `pdb` commands:
# ![pdb-cheatsheet](pdb-cheatsheet.png)
# 

# ## What is pdb?
# `pdb` is a debugger that is included in python's standard library. With a debugger, you can:
# 
# - Explore the state of a running program
# - Test implementation code before applying it
# - Follow the program's execution logic
# 
# You can set breakpoints at any point of your program to stop it and apply the three points above. Debuggers are very useful tools that can speed up the debugging process rather than using print() statements everywhere.

# ## Getting started with pdb
# Since `pdb` is included in python's standard library, we need to import it as we would any other python standard library. It works out of the box with jupyter notebooks, so run the cell below to import it.

# In[1]:


import pdb


# Now that we have `pdb` imported, let's go through a simple example of how to use `pdb`. Below we have a function, factorial, which takes in an argument x and computes the factorial x!. There's nothing wrong with the code, but let's still use `pdb` to walk through the logic of the function.
# 
# We start by adding the following line to the function: `pdb.set_trace()`
# 
# The `set_trace()` function hard codes a breakpoint at line where the command is placed. In this function, it is placed at line 2.
# 
# If we run the cell below, you'll notice that the `pdb` becomes active and we can start poking around our code! For now, just enter the command `q` (quit) into the interactive command box and go onto the next section, **5 pdb commands**. The result of quitting looks a lot like an error, but don't worry about it, `pdb` is just handling the quit command.
# 
# Just a heads up that if you re-run a debugger cell without quitting/completing it, the kernel may crash. Make sure to quit or exit the function before running other cells.

# In[2]:


def factorial(x):             # line 1
    pdb.set_trace()           # line 2
    if x == 1:                # line 3
        return 1              # line 4
    return x * factorial(x-1) # line 5
factorial(3)                  # line 6


# ## 5 pdb commands
# Taken from the `pdb` documentation, here are 6 commands that you'll find yourself using most frequently / are most useful in debugging your code.
# 
# 1. `l(ist)` - Displays 11 lines around the current line or continue the previous listing.
# 2. `s(tep)` - Execute the current line, stop at the first possible occasion.
# 3. `n(ext)` - Continue execution until the next line in the current function is reached or it returns.
# 4. `b(reak)` - Set a breakpoint (depending on the argument provided).
# 5. `r(eturn)` - Continue execution until the current function returns.
# 6. `h(elp)` - Without argument, print the list of available commands. With a command as an argument, print help about that command.
# 
# Note the parentheses around the last part of each command. These parentheses mean that they are _optional_ when using the command in `pdb`. However, if you have a variable in your code (such as `l` or `n`) then the `pdb` command takes precedence. So if you have a variable named `c` and you type `c` into `pdb`, you'll be issuing the `c(ontinue)` command which executes the program until it hits a breakpoint. Not exactly what we wanted right? A good practice in general is to give more informative variable names.
# 
# For the rest of the tutorial, I will be using the shortened version of the commands and if I use a command that I have not introduced here, I will explain what it does. So, let's begin with the first one.

# ### 1. l(ist) a.k.a. I'm too lazy to open the file containing the source code
# 
# ```
# l(ist) [first [,last]]
#     List source code for the current file. Without arguments, list 11 lines around the current line
#     or continue the previous listing. With one argument, list 11 lines starting at that line.
#     With two arguments, list the given range; if the second argument is less than the first, it is a count.
# ```
# 
# **NB**: The above description was generated by calling `help` on `list`. To get the same output, in the `pdb` REPL type `help l`.
# 
# `list` can help us examine the source code of the current file we are in. `list` allows you to specify a given range of lines you wish to see, which can be helpful. Note that the command `ll` (long list) shows you the source code for the current function or frame. This is often better than using `l`, since it's better knowing which function you are in rather than the random 11 lines around your current position. Let's try using `l` now!
# 
# Run the cell below and enter `l` into the interactive command box. Notice that we printed out the entire function. Now let's enter `l 2,4` to see lines 2 through 4. You'll see the lines 2 through 4 displayed in the session. Now let's run `ll` to see the entire function. Note that this output is the same as entering `l`, but if we had a longer function `ll` would display the whole function while `l` would only display 11 lines around your current line. To exit the current pdb session enter `q` (quit). Make sure to do this before running the cell again, otherwise you'll get stuck and have to restart your kernel in order to regain control. 

# In[3]:


factorial(3)


# ### 2. `s(tep)` a.k.a let's see what this method does...
# 
# ```
# s(tep)
#     Execute the current line, stop at the first possible occasion
#     (either in a function that is called or in the current
#     function).
# ```
# 
# Let's see what the `s` command does. Run the cell below and enter `s`. You'll notice that `s` will step through your function line by line, entering function calls made along the way. From the breakpoint, if you enter `s` 3 times you'll get the following output:
# 
# 1. `if x == 1:                # line 3`
# 2. `return x * factorial(x-1) # line 5`
# 3. `def factorial(x):         # line 1`
# 
# What if you enter `s` 3 more times? We end up stepping into the `set_trace()` function. We don't really need to know how this works, so enter `c` to continue until the next break point. Since we're recursively calling `factorial(x)`, we'll need to step through a few more times. Enter `c` until we exit the function and return the value of 3!, 6 or enter `q` to quit.

# In[4]:


factorial(3)


# ### 3. `n(ext)` a.k.a I hope this current line doesn't throw an exception
# 
# ```
# n(ext)
#     Continue execution until the next line in the current function
#     is reached or it returns.
# ```
# 
# Run the cell below and type the `n(ext)` command followed by `list` (notice a pattern) and let's observe what happens. This is similar to `s`, except we always continue to the next line in the **current** function, meaning we won't step into other functions. Keep entering `n` until we get some weird output. Note that when we get to the final `return` statement, we'll get the following output:
# 
# `/anaconda3/envs/data100/lib/python3.6/site-packages/IPython/core/displayhook.py(247)__call__()
# -> def __call__(self, result=None):`
# 
# This shows some of the internals of jupyter notebook and it's safe to ignore. Let's enter `c` to exit this session.

# In[5]:


factorial(3)


# ### 4. `b(reak)` a.k.a I don't want to type `n` anymore
# 
# ```
# b(reak) [ ([filename:]lineno | function) [, condition] ]
#     Without argument, list all breaks.
# 
#     With a line number argument, set a break at this line in the
#     current file.  With a function name, set a break at the first
#     executable line of that function.  If a second argument is
#     present, it is a string specifying an expression which must
#     evaluate to true before the breakpoint is honored.
# 
#     The line number may be prefixed with a filename and a colon,
#     to specify a breakpoint in another file (probably one that
#     hasn't been loaded yet).  The file is searched for on
#     sys.path; the .py suffix may be omitted.
# ```
# 
# We're only looking at the first two paragraphs of the `b(reak)` description in this tutorial. Let's look at a new function, `iterative_factorial(x)` which computes the factorial of x iteratively. Say we wanted to compute `iterative_factorial(20)`. Instead of looping through the code using `n` we can set a break point past the `while` loop and continue through until the breakpoint. Let's enter `b 7` to set a breakpoint on line 7, `return fac`. Then enter `c` to the break point that we just set. Nice! Instead of entering `n` a bunch of times we were able to directly continue past the `while` loop. Hit `c` again to exit. 
# 
# The documentation for `b(reak)` looks confusing. Here's a few more example usages and corresponding explanations:
# 
# Command | Explanation
# --- | ---
# `b` | view all set breakpoints
# `b 7` | set a breakpoint on line 7
# `b util:5` | set a breakpoint in `util.py` on line 5
# `b util.get_path` | set a breakpoint in `util.py` at the `get_path` function
# `b util.get_path, not filename.startswith('/')` | set a breakpoint in `util.py` at the `get_path` function that breaks if `filename` doesn't start with `/` 
# `cl 1` | clear breakpoint 1
# `cl util:5` | clear the breakpoint in `util.py` on line 5
# 
# 
# There's a lot more you can do with this command, so please check the tutorial for further instructions.

# In[6]:


def iterative_factorial(x): # Line 1
    pdb.set_trace()         # Line 2
    fac = 1                 # Line 3
    while x > 0:            # Line 4
        fac *= x            # Line 5
        x -= 1              # Line 6
    return fac              # Line 7


# In[7]:


iterative_factorial(10)


# ### 5. `r(eturn)` a.k.a. I want to get out of this function
# 
# ```
# r(eturn)
#     Continue execution until the current function returns.
# ```
# 
# The `return` is a great _power user_ command that let's you examine the final outcome of a function. While you could set a breakpoint at the return call, the `return` pdb command will help if there are multiple return statements in a single function since it only follows the path of execution for a single return. Let's call the `return` command and get to the end of the function. This immediately jumps to the `return` line and entering `c` will exit the function.

# In[8]:


iterative_factorial(10)


# ## Summary
# Congratulations! You now know the basics of `pdb`. This will be an extremeley useful tool in not only this class, but your future classes and your career (if you are writing code). There is a slight learning curve, but it's a worthwhile investment to working smarter and more efficiently. We strongly encourage you to try out `pdb` for this course and challenge yourself to use better techniques for writing and debugging code. You'll thank yourself later for doing this (and so will the GSI's `;)` )

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
