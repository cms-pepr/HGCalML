# Create the file ~/private/ml4reco_sql.config with your credentials (shared by email)
# IT should have the format:
# [settings]
# password : XXXX
# username : user_xxx
# host : XX.XXX.XX.XXX
# database : database_XX 


print("MODULE OBSOLETE?",__name__)

import configparser
import os

config = configparser.ConfigParser()
configfile = os.path.expanduser("~/private/ml4reco_sql.config")
if not os.path.isfile(configfile):
    credentials = -1
else:
    config.read(configfile)

    credentials = dict(config['settings'])

    if not all(credentials.values()):
        raise NotImplementedError("Set  username, password, etc in this file before proceeding.")
