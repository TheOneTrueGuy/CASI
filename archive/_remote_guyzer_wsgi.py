# Activate virtual environment
import os
import sys

# Add the virtual environment site-packages to the path
venv_path = "/home/Guyzer/.virtualenvs/myvirtualenv"
site_packages = os.path.join(venv_path, "lib", "python3.10", "site-packages")
sys.path.insert(0, site_packages)

# Also add the virtual environment bin directory to the path
bin_path = os.path.join(venv_path, "bin")
os.environ["PATH"] = bin_path + ":" + os.environ.get("PATH", "")
# This file contains the WSGI configuration required to serve up your
# web application at http://<your-username>.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.
#
# The below has been auto-generated for your Flask project


#import logging
import os
#os.environ['RUNPOD_API_KEY'] = 'test_random_value_12345'
#logging.warning("RUNPOD_API_KEY at app startup: %r", os.environ.get("RUNPOD_API_KEY"))

import re
import sys


#output = os.popen('pip freeze').read()

# Open a file for writing and write the output to it
#with open('/home/Guyzer/freeze.txt', 'w') as f:
#    f.write(output)

# add your project directory to the sys.path
#abpath='/home/Guyzer/Flask-AppBuilder'
project_home = '/home/Guyzer/Firsty'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path
#if abpath not in sys.path:
#    sys.path = [abpath] + sys.path

#sys.path = [re.sub('python3\.8', 'python3.9', path) for path in map(str, sys.path)]
#sys.path = [path.replace('python38.zip', 'python39.zip') for path in sys.path]
#with open('/home/Guyzer/syspath.txt', 'w') as f:
#    f.write(str(sys.path))




# import flask app but need to call it "application" for WSGI to work
from app import app as application  # noqa

# Forcing a hard reload by modifying the WSGI file.
# Reload trigger.....
# Force restart Thu Sep 25 23:54:52 UTC 2025
