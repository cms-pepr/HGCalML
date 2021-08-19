import mysql.connector
import os
from datetime import datetime, timezone
import threading
from queue import Queue
import numpy as np


class DValueError(ValueError):
    pass


_debug_mode = False

class ExperimentDatabaseManager():
    def __init__(self, mysql_credentials, cache_size = 100):

        self.mysql_credentials = mysql_credentials

        con, cur = self.connect()

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

        self.experiment_name = ''

        self.data_tables_verified = set()
        self.data_queue = Queue()
        self.cache_size = cache_size
        self.pusher_queue = Queue()

        t = threading.Thread(target=self._data_pusher_thread, args=(self.pusher_queue,))
        t.start()
        self.pusher_thead = t

    def _data_pusher_thread(self, queue):
        while True:
            try:
                data_to_be_pushed = queue.get()  # 3s timeout
                if data_to_be_pushed is None:
                    # End of thread
                    break

                con, cur = self.connect()

                for table_name, data, is_array in data_to_be_pushed:
                    if is_array:
                        query = 'insert into %s (experiment_name' % table_name
                        # values_query_section = '(\'%s\'' % (self.experiment_name)

                        values = []
                        for key, value in data.items():
                            query += ', %s' % key
                            values.append(value.tolist() if type(value) is np.ndarray else value)
                        values = [[(v if np.isfinite(v) else 0) if type(v) is float else v for v in value_array] for value_array in values]

                        query += ') values\n'
                        values_query_section = ['(\'' +self.experiment_name+'\',' + ','.join(['\''+str(s) +'\'' for s in vtuple])+')' for vtuple in zip(*values)]
                        values_query_section = ',\n'.join(values_query_section)

                        query += values_query_section
                    else:
                        query = 'INSERT INTO %s (experiment_name' % table_name

                        values_query_section = '(\'%s\'' % (self.experiment_name)
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

            except queue.Empty:
                continue

    def connect(self):
        con = mysql.connector.connect(
            host=self.mysql_credentials['host'],
            user=self.mysql_credentials['username'],
            password=self.mysql_credentials['password'],
            database=self.mysql_credentials['database'],
        )

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
        con, cur = self.connect()
        datatype = self._get_type(field_example)
        query = 'ALTER TABLE %s ADD %s %s;' % (table_name, field_name, datatype)
        con.execute(query)
        con.commit()
        con.close()

    def _verify_data_table(self, table_name, data):
        if table_name not in self.data_tables_verified:
            con, cur = self.connect()
            query = "SELECT table_name from information_schema.tables WHERE table_name='%s';" % table_name

            # query = "SELECT name from sqlite_master WHERE type='table' AND name='" + table_name + "'"
            cur.execute(query)
            result = cur.fetchall()

            if len(result) == 0:
                query = 'CREATE TABLE %s (' % table_name
                query += 'experiment_name text NOT NULL'

                for key, value in data.items():
                    # if type(value) is np.ndarray:
                    #     if len(value) == 0:
                    #         con.commit()
                    #         con.close()
                    #         raise ValueError('Array length is zero')
                    #     value = value[0].item()
                    #
                    #     if type(value) is list:
                    #         if len(value) == 0:
                    #             con.commit()
                    #             con.close()
                    #             raise ValueError('Array length is zero')
                    #         value = value[0]
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


                cur.execute(
                    '''CREATE INDEX idx_%s ON %s(experiment_name(300));''' % (table_name, table_name))

                cur.execute("SELECT * from data_tables WHERE data_table_name='%s'" % (table_name))
                rows = cur.fetchall()
                if len(rows) ==0:
                    cur.execute("INSERT INTO data_tables VALUES ('%s')" % (table_name))

            self.data_tables_verified.add(table_name)
            con.commit()
            con.close()

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
            self.pusher_queue.put(data_to_be_pushed)

    def close(self):
        data_to_be_pushed = []
        while self.data_queue.qsize() > 0:
            table_name, data, is_array = self.data_queue.get()
            data_to_be_pushed.append([table_name, data, is_array])
        if len(data_to_be_pushed) > 0:
            self.pusher_queue.put(data_to_be_pushed)
        self.pusher_queue.put(None)
        self.pusher_thead.join()

    def flush(self):
        data_to_be_pushed = []
        while self.data_queue.qsize() > 0:
            table_name, data, is_array = self.data_queue.get()
            data_to_be_pushed.append([table_name, data, is_array])
        if len(data_to_be_pushed) > 0:
            self.pusher_queue.put(data_to_be_pushed)

    def insert_experiment(self, name):
        con, cur = self.connect()
        command = "INSERT INTO experiments VALUES ('%s','%s')" % (name, datetime.now(timezone.utc))
        if _debug_mode:
            print(command)
        cur.execute(command)
        con.commit()
        con.close()

    def experiment_exists(self, name):
        con, cur = self.connect()
        cur.execute("SELECT * FROM experiments WHERE experiment_name='%s';" % (name))
        rows = cur.fetchall()
        len_of_rows = len(rows) > 0
        con.commit()
        con.close()
        return len_of_rows > 0

    def delete_experiment(self, experiment_name):
        con, cur = self.connect()

        query = "SELECT data_table_name from data_tables"
        cur.execute(query)
        result = cur.fetchall()
        for table_name in result:
            table_name = table_name[0]
            cur.execute("DELETE FROM %s WHERE experiment_name='%s';" % (table_name, experiment_name))


            cur.execute("SELECT * FROM %s" % (table_name))
            rows = cur.fetchall()
            if len(rows) == 0:
                # Delete the table as well as well as its reference from data_tables table
                cur.execute("DELETE FROM data_tables WHERE data_table_name='%s';" % (table_name))
                cur.execute('DROP TABLE %s' % table_name)

        cur.execute("DELETE FROM experiments WHERE experiment_name='%s';" % (experiment_name))
        con.commit()

        con.close()
