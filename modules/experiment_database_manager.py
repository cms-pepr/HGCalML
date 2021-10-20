import sqlite3
import traceback

import mysql.connector
import os
from datetime import datetime, timezone
import threading
from queue import Queue
import numpy as np
import time


class DValueError(ValueError):
    pass


_debug_mode = False


class ExperimentDatabaseManager():
    class DataPusherThread(threading.Thread):
        def __init__(self, queue, database_manager_class, is_mysql=False, is_file=False):
            threading.Thread.__init__(self)

            if is_mysql and is_file:
                raise RuntimeError("Error...")
            if not is_mysql and not is_file:
                raise RuntimeError("Error...")

            self.is_mysql = is_mysql
            self.is_file = is_file
            self.queue = queue
            self.database_manager_object=database_manager_class

        def run(self):
            get_new_element = True
            data_to_be_pushed = None
            while True:
                try:
                    if self.is_mysql:
                        con, cur = self.database_manager_object.connect_to_mysql()
                    else:
                        con, cur = self.database_manager_object.connect_to_file()

                    if get_new_element or data_to_be_pushed is None:
                        data_to_be_pushed = self.queue.get()  # 3s timeout
                    else:
                        get_new_element = True

                    if data_to_be_pushed is None:
                        # End of thread
                        break

                    for table_name, data, is_array in data_to_be_pushed:
                        if is_array:
                            query = 'insert into %s (experiment_name' % table_name
                            # values_query_section = '(\'%s\'' % (self.experiment_name)

                            values = []
                            for key, value in data.items():
                                query += ', %s' % key
                                values.append(value.tolist() if type(value) is np.ndarray else value)
                            values = [[(v if np.isfinite(v) else 0) if type(v) is float else v for v in value_array] for
                                      value_array in values]

                            query += ') values\n'
                            values_query_section = ['(\'' + self.database_manager_object.experiment_name + '\',' + ','.join(
                                ['\'' + str(s) + '\'' for s in vtuple]) + ')' for vtuple in zip(*values)]
                            values_query_section = ',\n'.join(values_query_section)

                            query += values_query_section
                        else:
                            query = 'INSERT INTO %s (experiment_name' % table_name

                            values_query_section = '(\'%s\'' % (self.database_manager_object.experiment_name)
                            for key, value in data.items():
                                if type(value) is float:
                                    value = value if np.isfinite(value) else 0.0
                                query += ', %s' % key
                                values_query_section += ', \'%s\'' % (value)
                            #
                            query += ') VALUES '
                            query += values_query_section + ')'

                        if _debug_mode:
                            print(query)
                        cur.execute(query)

                    con.commit()
                    con.close()

                except Exception as e:
                    if self.is_mysql:
                        if type(e) == mysql.connector.errors.InterfaceError or type(
                                mysql.connector.errors.OperationalError):
                            print("Error connecting to server, will try again in a second")
                            print(query)

                            time.sleep(1)
                            if con.is_connected():
                                con.close()
                            get_new_element = False
                            continue
                    print(e.args)
                    print(e)
                    traceback.print_exc()



    def __init__(self, mysql_credentials=None, file=None, cache_size = 100):
        if mysql_credentials==-1:
            raise NotImplementedError("MySQL credentials not set, follow instructions in sql_credentials.py to set them up.")

        if file == None and mysql_credentials == None:
            raise RuntimeError("Both file and mysql credentials are None. No where to store data")
        # if file != None and mysql_credentials != None:
        #     raise RuntimeError("Can't set both file and mysql. Choose one.")

        self.mysql_credentials = mysql_credentials
        self.file = file

        self.has_mysql = self.mysql_credentials != None
        self.has_file = self.file != None


        if self.has_mysql:
            con, cur = self.connect_to_mysql()
            query = "SELECT table_name from information_schema.tables WHERE table_name='experiments';"
            cur.execute(query)
            result = cur.fetchall()

            if len(result) == 0:
                cur.execute('''CREATE TABLE experiments
                               (experiment_name text NOT NULL, date_started DATETIME)''')
                cur.execute('''CREATE UNIQUE INDEX idx_training ON experiments (experiment_name(300));''')

                cur.execute('''CREATE TABLE data_tables
                               (data_table_name text)''')
                cur.execute('''CREATE UNIQUE INDEX idx_data_table_name ON data_tables (data_table_name(300));''')

                con.commit()

            con.close()
        if self.has_file:
            con, cur = self.connect_to_file()

            query = "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments';"
            cur.execute(query)
            result = cur.fetchall()
            if len(result) == 0:
                cur.execute('''CREATE TABLE experiments
                               (experiment_name text NOT NULL, date_started DATETIME)''')
                cur.execute('''CREATE UNIQUE INDEX idx_training ON experiments (experiment_name);''')

                cur.execute('''CREATE TABLE data_tables
                               (data_table_name text)''')
                cur.execute('''CREATE UNIQUE INDEX idx_data_table_name ON data_tables (data_table_name);''')

                con.commit()

            con.close()


        self.experiment_name = ''

        self.data_tables_verified = set()
        self.data_queue = Queue()
        self.cache_size = cache_size

        if self.has_mysql:
            self.mysql_pusher_queue = Queue()
            t = ExperimentDatabaseManager.DataPusherThread(self.mysql_pusher_queue, self, is_mysql=True)
            t.start()
            self.mysql_pusher_thread = t

        if self.has_file:
            self.file_pusher_queue = Queue()
            t = ExperimentDatabaseManager.DataPusherThread(self.file_pusher_queue, self, is_file=True)
            t.start()
            self.file_pusher_thread = t


    def connect_to_mysql(self):
        if not self.has_mysql:
            raise RuntimeError("Not saving to mysql. Try using connect_to_file instead.")
        con = mysql.connector.connect(
            host=self.mysql_credentials['host'],
            user=self.mysql_credentials['username'],
            password=self.mysql_credentials['password'],
            database=self.mysql_credentials['database'],
        )

        cur = con.cursor()

        return con, cur
    def connect_to_file(self):
        if not self.has_file:
            raise RuntimeError("Not saving to file. Try using connect_to_mysql instead.")
        con = sqlite3.connect(self.file)
        cur = con.cursor()

        return con, cur

    def connect(self):
        print("Function obselete... use connect_to_mysql or connect_to_file instead.")
        if self.has_mysql:
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

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name
        if not self.experiment_exists(self.experiment_name):
            if _debug_mode:
                print("Experiment doesn't exist, creating one")
            self.insert_experiment(self.experiment_name)
        else:
            if _debug_mode:
                print("Experiment  already exists")

    def _get_type(self, field_example):
        if _debug_mode:
            print("Getting type", field_example, type(field_example), type(field_example) is float)
        error_string = "Unknown data type, please try casting to float, int or string before writing for safety reasons.\n" \
                      "Multi dimensional arrays are not supported\n"\
                      "Another possible reason is you are using numpy scalars. Please don't use numpy scalars"

        if type(field_example) is np.ndarray or type(field_example) is list:
            if np.isscalar(field_example):
                raise DValueError(error_string)
            if len(field_example) == 0:
                print("Zero length array detected")
                raise DValueError('Zero length array')
            if type(field_example) is np.ndarray:
                field_example = field_example[0].item()
            else:
                field_example = field_example[0]

        if _debug_mode:
            print(type(field_example))

        if type(field_example) is float:
            typex = 'real'
        elif type(field_example) is int:
            typex = 'bigint'
        elif type(field_example) is str:
            typex = 'text'
        else:
            raise DValueError()
        return typex

    def add_another_field_to_experiment_data(self, table_name, field_name, field_example):
        datatype = self._get_type(field_example)
        query = 'ALTER TABLE %s ADD %s %s;' % (table_name, field_name, datatype)

        if self.has_mysql:
            con, cur = self.connect_to_mysql()
            con.execute(query)
            con.commit()
            con.close()

        if self.has_file:
            con, cur = self.connect_to_file()
            con.execute(query)
            con.commit()
            con.close()

    def _verify_data_table(self, table_name, data):
        def work(for_mysql):
            if table_name not in self.data_tables_verified:
                if for_mysql:
                    con, cur = self.connect_to_mysql()
                else:
                    con, cur = self.connect_to_file()

                if for_mysql:
                    query = "SELECT table_name from information_schema.tables WHERE table_name='%s';" % table_name
                else:
                    query = "SELECT name FROM sqlite_master WHERE type='table' AND name='%s';" % table_name

                # query = "SELECT name from sqlite_master WHERE type='table' AND name='" + table_name + "'"
                cur.execute(query)
                result = cur.fetchall()

                if len(result) == 0:
                    query = 'CREATE TABLE %s (' % table_name
                    query += 'experiment_name text NOT NULL'

                    for key, value in data.items():
                        try:
                            typex = self._get_type(value)
                        except DValueError:
                            con.commit()
                            con.close()
                            raise ValueError('Error occured in getting data type\n'
                                             'Supported types are long, double and string.\n'
                                             'Could also be list/numpy array of size zero.')
                        query += ', %s %s' % (key, typex)

                    query += ');'
                    # print(query)
                    cur.execute(query)

                    if for_mysql:
                        query = '''CREATE INDEX idx_%s ON %s(experiment_name(300));''' % (table_name, table_name)
                    else:
                        query = '''CREATE INDEX idx_%s ON %s(experiment_name);''' % (table_name, table_name)
                    cur.execute(query)

                    cur.execute("SELECT * from data_tables WHERE data_table_name='%s'" % (table_name))
                    rows = cur.fetchall()
                    if len(rows) == 0:
                        cur.execute("INSERT INTO data_tables VALUES ('%s')" % (table_name))

                con.commit()
                con.close()

        if self.has_mysql:
            work(for_mysql=True)
        if self.has_file:
            work(for_mysql=False)
        self.data_tables_verified.add(table_name)



    def insert_experiment_data(self, table_name, data):
        if len(self.experiment_name) == 0:
            raise RuntimeError("Initiate database class using a training name")

        self._verify_data_table(table_name, data)

        is_array = False
        for key, value in data.items():
            if type(value) is np.ndarray or type(value) is list:
                is_array = True
                break

        self.data_queue.put((table_name, data, is_array))

        if self.data_queue.qsize() > self.cache_size:
            data_to_be_pushed = []
            while self.data_queue.qsize() > 0:
                table_name, data, is_array = self.data_queue.get()
                data_to_be_pushed.append([table_name, data, is_array])

            if self.has_mysql:
                self.mysql_pusher_queue.put(data_to_be_pushed)
            if self.has_file:
                self.file_pusher_queue.put(data_to_be_pushed)

    def close(self):
        data_to_be_pushed = []
        while self.data_queue.qsize() > 0:
            table_name, data, is_array = self.data_queue.get()
            data_to_be_pushed.append([table_name, data, is_array])
        if len(data_to_be_pushed) > 0:
            if self.has_mysql:
                self.mysql_pusher_queue.put(data_to_be_pushed)
            if self.has_file:
                self.file_pusher_queue.put(data_to_be_pushed)

        if self.has_mysql:
            self.mysql_pusher_queue.put(None)
            self.mysql_pusher_thread.join()

        if self.has_file:
            self.file_pusher_queue.put(None)
            self.file_pusher_thread.join()

    def flush(self):
        data_to_be_pushed = []
        while self.data_queue.qsize() > 0:
            table_name, data, is_array = self.data_queue.get()
            data_to_be_pushed.append([table_name, data, is_array])
        if len(data_to_be_pushed) > 0:
            if self.has_mysql:
                self.mysql_pusher_queue.put(data_to_be_pushed)
            if self.has_file:
                self.file_pusher_queue.put(data_to_be_pushed)

    def insert_experiment(self, name):
        query = "INSERT INTO experiments VALUES ('%s','%s')" % (name, datetime.now(timezone.utc))
        if self.has_mysql:
            con, cur = self.connect_to_mysql()
            cur.execute(query)
            con.commit()
            con.close()
        if self.has_file:
            con, cur = self.connect_to_file()
            cur.execute(query)
            con.commit()
            con.close()

    def experiment_exists(self, name):
        query = "SELECT * FROM experiments WHERE experiment_name='%s';" % (name)
        if self.has_mysql:
            con, cur = self.connect_to_mysql()
        else:
            con, cur = self.connect_to_file()

        cur.execute(query)
        rows = cur.fetchall()
        len_of_rows = len(rows) > 0
        con.commit()
        con.close()
        return len_of_rows > 0


    def _delete_experiment(self, con, cur, experiment_name, ismysql=False):
        query = "SELECT data_table_name from data_tables"
        cur.execute(query)
        result = cur.fetchall()
        if ismysql:
            query = "SELECT table_name from information_schema.tables"
        else:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments';"
        cur.execute(query)
        result2 = cur.fetchall()

        table_names_1 = [x[0] for x in result]
        table_names_2 = [x[0] for x in result2]

        table_names = []
        for t in table_names_1:
            if t in table_names_2:
                table_names.append(t)

        for table_name in table_names:
            query = "DELETE FROM %s WHERE experiment_name='%s';" % (table_name, experiment_name)
            cur.execute(query)


            query = "SELECT count(experiment_name) FROM %s" % (table_name)
            cur.execute(query)
            rows = cur.fetchall()

            if rows[0][0] == 0:
                # Delete the table as well as well as its reference from data_tables table
                cur.execute("DELETE FROM data_tables WHERE data_table_name='%s';" % (table_name))
                cur.execute('DROP TABLE %s' % table_name)

        cur.execute("DELETE FROM experiments WHERE experiment_name='%s';" % (experiment_name))
        con.commit()



    def delete_experiment(self, experiment_name):
        if self.has_mysql:
            con, cur = self.connect_to_mysql()
            self._delete_experiment(con, cur, experiment_name, ismysql=True)
            con.close()
        if self.has_file:
            con, cur = self.connect_to_file()
            self._delete_experiment(con, cur, experiment_name, ismysql=False)
            con.close()
