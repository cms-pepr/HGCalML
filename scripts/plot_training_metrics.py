import sql_credentials
import experiment_database_reading_manager
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def running_mean(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


parser = argparse.ArgumentParser(
    'Produce running metrics plot (loss, efficiency and more)')
parser.add_argument('experiment_name',
                    help='Experiment name')
parser.add_argument('output',
                    help='PDF file')


args = parser.parse_args()



manager = experiment_database_reading_manager.ExperimentDatabaseReadingManager(mysql_credentials=sql_credentials.credentials)
print("Getting data now")
# args.exper
training_performance_metrics = manager.get_data('training_performance_metrics_extended', experiment_name=args.experiment_name)
print("Received data")


average_over = 10

pdf = PdfPages(args.output)
training_performance_metrics['efficiency'] = [float(x) for x in training_performance_metrics['efficiency']]
plt.figure()
eff = training_performance_metrics['efficiency']
eff = running_mean(eff, average_over)
plt.plot(training_performance_metrics['iteration'], eff)
plt.xlabel('iteration')
plt.ylabel('efficiency')
pdf.savefig()


training_performance_metrics['fake_rate'] = [float(x) for x in training_performance_metrics['fake_rate']]

plt.figure()
fak = training_performance_metrics['fake_rate']
fak = running_mean(fak, average_over)
plt.plot(training_performance_metrics['iteration'], fak)
plt.xlabel('iteration')
plt.ylabel('fake_rate')
pdf.savefig()

training_performance_metrics['sum_response'] = [float(x) for x in training_performance_metrics['sum_response']]

plt.figure()
sres = training_performance_metrics['sum_response']
sres = running_mean(sres, average_over)
plt.plot(training_performance_metrics['iteration'], sres)
plt.xlabel('iteration')
plt.ylabel('response')
pdf.savefig()



plt.figure()
loss = training_performance_metrics['loss']
loss = running_mean(sres, average_over)
plt.plot(training_performance_metrics['iteration'], loss)
plt.xlabel('iteration')
plt.ylabel('loss')

pdf.savefig()

pdf.close()
