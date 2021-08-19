# Do not upload username/password to github
# Change your username, password, and host and then changed "fixed" variable to True

password='yF!8&cU3QXe8'
username='user_klong'
host = '31.164.89.251'
database = 'database_h1'

credentials = {'host':host, 'password':password, 'username':username, 'database':database}

fixed = True


if not fixed:
    raise NotImplementedError("Set  username, password, etc in this file before proceeding.")
