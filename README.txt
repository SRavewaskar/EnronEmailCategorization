Topic: Automatic email classification and categorization into organized bundles.
Author: Saurabh Rewaskar

README.txt: This file contains the description of the contents of this project and how to execute the program
data/ : The data folder
categories.txt(http://bailando.sims.berkeley.edu/enron/enron_categories.txt):
The categories which are used to label the emails
classify.py: The main program, this is the program which needs to be executed.

How to run this project:
1] Setting up the environment.
Requires Python 3.6.
Libraries required:
pandas
numpy
scikit-learn
timeit
(Alternatively use the Anaconda Data Science platform)

2] The data is to be downloaded from the link:
http://bailando.sims.berkeley.edu/enron/enron_with_categories.tar.gz
and extracted into the data/ folder.
The data directory should look like this after data extraction into it.
data/
  enron_with_categories/
    1/
    2/
    3/
    4/
    5/
    6/
    7/
    8/

3] Run the script classify.py
Expected time to complete ~15min
