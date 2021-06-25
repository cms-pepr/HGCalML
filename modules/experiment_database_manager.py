import sqlite3
import os
from datetime import datetime, timezone
import threading
import mysql.connector

class ExperimentDatabaseManager():
    def __init__(self, mysql_credentials):

        self.mysql_credentials = mysql_credentials

        self.con = mysql.connector.connect(
            host=mysql_credentials['host'],
            user=mysql_credentials['username'],
            password=mysql_credentials['password'],
            database=mysql_credentials['database'],
        )

        # file_path_exists = os.path.exists(filepath)
        # self.con = sqlite3.connect(filepath, check_same_thread=False)
        self.cur = self.con.cursor()
        self.lock = threading.Lock()

        with self.lock:

            query = "SELECT table_name from information_schema.tables WHERE table_name='experiments';"
            self.cur.execute(query)
            result = self.cur.fetchall()

            if len(result) == 0:
                    self.cur.execute('''CREATE TABLE experiments
                                   (experiment_name text NOT NULL, date_started DATETIME, training_path text)''')
                    self.cur.execute('''CREATE UNIQUE INDEX idx_training ON experiments (experiment_name(10));''')

                    self.cur.execute('''CREATE TABLE data_tables
                                   (data_table_name text)''')
                    self.cur.execute('''CREATE UNIQUE INDEX idx_data_table_name ON data_tables (data_table_name(10));''')

                    self.con.commit()

        self.experiment_name = ''

    def set_experiment(self, experiment_name, experiment_path):
        self.experiment_name = experiment_name
        if not self.experiment_exists(self.experiment_name):
            print("Training doesn't exist, creating one")
            self.insert_experiment(self.experiment_name, experiment_path)
        else:
            print("Training already exists")

    def add_another_field_to_experiment_data(self, table_name, field_name, field_example):
        with self.lock:
            datatype = 'text' if type(field_example) is not float else 'float'
            query = 'ALTER TABLE %s ADD %s %s;' % (table_name, field_name, datatype)
            self.con.execute(query)
            self.con.commit()

    def insert_experiment_data(self, table_name, data):
        if len(self.experiment_name) == 0:
            raise RuntimeError("Initiate database class using a training name")

        with self.lock:
            query = "SELECT table_name from information_schema.tables WHERE table_name='%s';" % table_name
            # query = "SELECT name from sqlite_master WHERE type='table' AND name='" + table_name + "'"
            self.cur.execute(query)
            result = self.cur.fetchall()

            if len(result) == 0:
                query = 'CREATE TABLE %s (' % table_name
                query += 'experiment_name text NOT NULL'

                for key, value in data.items():
                    typex = 'text'
                    if type(value) is float:
                        typex = 'real'
                    query += ', %s %s' % (key, typex)

                query += ');'
                self.cur.execute(query)
                self.cur.execute(
                    '''CREATE INDEX idx_experiment_name_%s ON %s(experiment_name(10));''' % (table_name, table_name))

                self.cur.execute("INSERT INTO data_tables VALUES ('%s')" % (table_name))

            if True:
                query = 'INSERT INTO %s (experiment_name' % table_name


                values_query_section = '(\'%s\'' % (self.experiment_name)
                for key, value in data.items():
                    query += ', %s' % key
                    values_query_section += ', \'%s\'' % (value)
                #
                query += ') VALUES '
                query += values_query_section + ')'
                self.cur.execute(query)

            self.con.commit()

    def close(self):
        with self.lock:
            self.con.commit()
            self.con.close()

    def insert_experiment(self, name, experiment_path):
        with self.lock:
            self.cur.execute(
                "INSERT INTO experiments VALUES ('%s','%s', '%s')" % (name, datetime.now(timezone.utc), experiment_path))
            self.con.commit()

    def experiment_exists(self, name):
        with self.lock:
            self.cur.execute("SELECT * FROM experiments WHERE experiment_name='%s';" % (name))
            rows = self.cur.fetchall()
            len_of_rows = len(rows) > 0
            self.con.commit()
            return len_of_rows > 0

    def delete_experiment(self, experiment_name):
        with self.lock:
            query = "SELECT data_table_name from data_tables"
            result = self.cur.execute(query)
            result = self.cur.fetchall()
            for table_name in result:
                table_name = table_name[0]
                self.cur.execute("DELETE FROM %s WHERE experiment_name='%s';" % (table_name, experiment_name))
                self.cur.execute("SELECT * FROM %s" % (table_name))
                rows = self.cur.fetchall()
                if len(rows) == 0:
                    # Delete the table as well as well as its reference from data_tables table
                    self.cur.execute("DELETE FROM data_tables WHERE data_table_name='%s';" % (table_name))
                    self.cur.execute('DROP TABLE %s' % table_name)

            self.cur.execute("DELETE FROM experiments WHERE experiment_name='%s';" % (experiment_name))
            self.con.commit()
