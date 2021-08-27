import sqlite3

import mysql.connector
import os
from datetime import datetime, timezone
import threading
from queue import Queue


_debug_mode = False


class ExperimentDatabaseReadingManager():
    def __init__(self, mysql_credentials=None, file=None):
        if mysql_credentials==-1:
            raise NotImplementedError("MySQL credentials not set, follow instructions in sql_credentials.py to set them up.")

        if file == None and mysql_credentials == None:
            raise RuntimeError("Both file and mysql credentials are None. No where to store data")
        if file != None and mysql_credentials != None:
            raise RuntimeError("Can't set both file and mysql. Choose one.")

        self.mysql_credentials = mysql_credentials
        self.file = file

        self.is_mysql = self.mysql_credentials!=None
        self.mysql_credentials = mysql_credentials

        self.field_names_cache = dict()


    def connect(self):
        if self.mysql_credentials:
            con = mysql.connector.connect(
                host=self.mysql_credentials['host'],
                user=self.mysql_credentials['username'],
                password=self.mysql_credentials['password'],
                database=self.mysql_credentials['database'],
            )

            cur = con.cursor()

            return con, cur
        else:
            con = sqlite3.connect(self.file)
            cur = con.cursor()
            return con, cur

        return con, cur

    def get_field_names(self, table_name):
        if _debug_mode:
            print("Getting field names")
        if table_name not in self.field_names_cache:
            con, cur = self.connect()
            if self.is_mysql:
                query = "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '%s';"%table_name
            else:
                query = "SELECT name FROM PRAGMA_TABLE_INFO('%s')" % table_name
            if _debug_mode:
                print(query)
            cur.execute(query)
            self.field_names_cache[table_name] = [x[0] for x in cur.fetchall()]


            con.close()

        return self.field_names_cache[table_name]

    def get_data_from_query(self, query):
        con, cur = self.connect()
        cur.execute(query)
        result = cur.fetchall()
        con.close()
        return result

    class TableDoesNotExistError(RuntimeError):
        pass


    def get_data(self, table_name, experiment_names=None, field_names=None, condition_string=None):
        query = "SELECT "
        if field_names is None:
            field_names = self.get_field_names(table_name)
        if len(field_names) == 0:
            raise ExperimentDatabaseReadingManager.TableDoesNotExistError('Table %s does not exist'%table_name)

        query += '%s ' % ','.join(field_names)

        query += 'FROM %s ' % table_name

        where_added = False
        if experiment_names is not None:
            if type(experiment_names) is list:
                if len(experiment_names) ==0:
                    raise RuntimeError("Length of experiment names is zero")
                query += "WHERE "
                operands = []
                for exp in experiment_names:
                    operands .append("experiment_name='%s'" % exp)
                query += ' (%s) '% ' OR '.join(operands)

            else:
                query += "WHERE experiment_name = '%s'" % experiment_names
            where_added=True

        if condition_string is not None:
            query += (' AND ' if where_added else ' WHERE ') + condition_string

        if _debug_mode:
            print(query)

        con, cur = self.connect()
        cur.execute(query)
        result = cur.fetchall()
        con.close()

        if len(result) == 0:
            return None


        result_as_dict = {}
        for i, field_name in enumerate(field_names):
            result_as_dict[field_name] = [x[i] for x in result]


        return result_as_dict
