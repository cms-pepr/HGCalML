# Do not upload username/password to github
# Change your username, password, and host and then changed "fixed" variable to True

password=''
username=''
host = ''
database = ''

credentials = {'host':host, 'password':password, 'username':username, 'database':database}

fixed = False


if not fixed:
    raise NotImplementedError("Set  username, password, etc in this file before proceeding.")
