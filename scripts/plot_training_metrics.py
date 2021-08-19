import sql_credentials
import experiment_database_reading_manager
import matplotlib.pyplot as plt
import numpy as np

manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
print("Getting data now")
training_performance_metrics = manager.get_data('training_performance_metrics', experiment_name='alpha_experiment_june_pca_double_cords_2')
print("Received data")


training_performance_metrics['efficiency'] = [float(x) for x in training_performance_metrics['efficiency']]
plt.plot(training_performance_metrics['iteration'], training_performance_metrics['efficiency'])
plt.xlabel('iteration')
plt.ylabel('efficiency')
plt.show()

training_performance_metrics['fake_rate'] = [float(x) for x in training_performance_metrics['fake_rate']]
plt.plot(training_performance_metrics['iteration'], training_performance_metrics['fake_rate'])
plt.xlabel('iteration')
plt.ylabel('fake_rate')
plt.show()


training_performance_metrics['sum_response'] = [float(x) for x in training_performance_metrics['sum_response']]



plt.plot(training_performance_metrics['iteration'], training_performance_metrics['sum_response'])
plt.xlabel('iteration')
plt.ylabel('response')
plt.show()


