#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import os

# Get absolute path of the actual Streamlit app
script_path = r"C:\Users\Jibran\yolov5\app12.py"
 

# Launch a new Command Prompt window and run the Streamlit app
command = f'start cmd /k streamlit run "{script_path}"'

subprocess.call(command, shell=True)

