# Create the file ~/private/ml4reco_sql.config with your credentials (shared by email)
# IT should have the format:
# [settings]
# password : XXXX
# username : user_xxx
# host : XX.XXX.XX.XXX
# database : database_XX 

import configparser
import os

config = configparser.ConfigParser()
configfile = os.path.expanduser("~/private/ml4reco_sql.config")
if not os.path.isfile(configfile):
    raise NotImplementedError(f"Please create the file '{configfile}' with your data base credentials added")

config.read(configfile)

credentials = dict(config['settings'])

if not all(credentials.values()):
    raise NotImplementedError("Set  username, password, etc in this file before proceeding.")
