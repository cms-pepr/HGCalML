import mysql.connector
import os
from datetime import datetime, timezone
import threading
from queue import Queue


_debug_mode = False


class ExperimentDatabaseReadingManager():
    def __init__(self, mysql_credentials):
        self.mysql_credentials = mysql_credentials

        self.field_names_cache = dict()


    def connect(self):
        con = mysql.connector.connect(
            host=self.mysql_credentials['host'],
            user=self.mysql_credentials['username'],
            password=self.mysql_credentials['password'],
            database=self.mysql_credentials['database'],
        )

        cur = con.cursor()

        return con, cur

    def get_field_names(self, table_name):
        if table_name not in self.field_names_cache:
            con, cur = self.connect()
            query = "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '%s';"%table_name
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


    def get_data(self, table_name, experiment_name=None, field_names=None):
        query = "SELECT "
        if field_names is None:
            field_names = self.get_field_names(table_name)


        query += '%s ' % ','.join(field_names)

        query += 'FROM %s ' % table_name

        if experiment_name is not None:
            query += "WHERE experiment_name = '%s'"%experiment_name

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
