###################################################################################
# trainertf.py: a simple trainer                                                  #
###################################################################################
# University of Applied Sciences Munich                                           #
# Dept of Electrical Enineering and Information Technology                        #
# Institute for Applications of Machine Learning and Intelligent Systems (IAMLIS) #
#                                                        (c) Alfred Schöttl 2022  #
###################################################################################
#                                                                                 #
# These classes give some basic functionality to train and evaluate               #
# networks.                                                                       #
#                                                                                 #
###################################################################################
###################################################################################
#  Modifided by Nicolas R.                                                        #
###################################################################################

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tqdm.notebook import trange
import matplotlib.pyplot as plt


class HistoryManager:

    ''' The HistoryManager provides basic functions to manage metric data which are computed
        during the training. The current training step is stored in overall_steps. The metric
        data consists of a <name> and a <val>, the name of test data should be test_<name>.
        - store data of current training step:  self._add_to_hist(<name>, <val>)
        - query data of current training step:  self.<name>
        - query current averaged data:          self.avg_<name>
        - query data for all steps:             self.get_hist(<name>) or self.get_avg_hist(<name>)
        - plot data for all steps:              self.plot_val(<name>)
    '''

    def __init__(self, config, steps_per_epoch_train, steps_per_epoch_test):
        self.hist            = None
        self.overall_steps   = 0
        self.avg_steps_train = config.get('avg_steps_train', steps_per_epoch_train)
        self.avg_steps_test  = config.get('avg_steps_test', steps_per_epoch_test)

    def _reset_hist(self):
        self.hist            = {}
        self.overall_steps   = 0

    def _add_to_hist(self, name, val):
        '''Adds a value to the history. The history is a dictionary. Each entry is itself a dictionary
           with the global step number as key.'''
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        if name not in self.hist:
            self.hist[name] = {}
        if self.overall_steps not in self.hist[name]:
            self.hist[name][self.overall_steps] = val
        else:
            self.hist[name][self.overall_steps] += val

    def _divide_hist_val_by(self, name, c):
        self.hist[name][self.overall_steps] /= c

    def get_curr(self, name):
        ''' Returns the current value of the history.'''
        if name in self.hist:
            return self.hist[name][self.overall_steps]
        else:
            raise NotImplementedError(f'Name {name} is not available.')

    def get_hist(self, name):
        ''' Returns the pair T, V of the metric name from the history. T, V are two vectors with time points and the values.'''
        if name in self.hist:
            return np.array(list(self.hist[name].keys())), np.array(list(self.hist[name].values()))
        else:
            raise NotImplementedError(f'Name {name} is not available.')

    def get_avg(self, name):
        ''' Returns the current averaged value of the history. The average is computed
            as moving average with a configurable window.'''
        if name in self.hist:
            n = self.avg_steps_test if name[:5] == 'test_' else self.avg_steps_train
            h = self.hist[name]
            h = [h[k] for k in
                 range(self.overall_steps-n+1 if self.overall_steps-n+1 > 0 else 0, self.overall_steps+1)]
            return np.array(h).mean()
        else:
            raise NotImplementedError(f'Name {name} is not available.')

    def get_avg_hist(self, name):
        ''' Returns the pair T, V of the averaged metric name from the history. T, V are two vectors with time points and the
            values. The average is computed as moving average with a configurable window.'''
        if name in self.hist:
            T, val = np.array(list(self.hist[name].keys())), np.array(list(self.hist[name].values()))
            n = min(self.avg_steps_test if name[:5] == 'test_' else self.avg_steps_train, len(val))
            cs = np.cumsum(val)
            smoothed_val_part1 = cs[:n] / np.arange(1,n+1)
            smoothed_val_part2 = ((cs[n:] - cs[:-n]) / n) if len(cs)>n else []
            return T, np.concatenate([smoothed_val_part1, smoothed_val_part2])
        else:
            raise NotImplementedError(f'Name {name} is not available.')

    def plot_val(self, name):
        ''' Plot a value from the metrics. The metric names may be prepended by "avg_" to obtain averaged versions.
            The average is computed as moving average with a configurable window.'''
        plt.gca().set_title(name)
        if name[:4] == 'avg_':  # compute the moving average of the history value
            T, val = self.get_avg_hist(name[4:])
            name_without_avg = name[4:]
        else:
            T, val = self.get_hist(name)
            name_without_avg = name
        plt.plot(T, val);
        if 'test_'+name_without_avg in self.hist and len(self.hist['test_'+name_without_avg]) > 0:
            if name[:4] == 'avg_':                   # compute the moving average of the history value
                T, val = self.get_avg_hist('test_' + name[4:])
            else:
                T, val = self.get_hist('test_' + name)
            plt.plot(T, val, 'r');
        plt.grid()

    def __getattr__(self, name):
        ''' This function allows access to the metrics' data by the syntax "self.metric_name", which is useful in the
            report function. The metric names may be prepended by "avg_" to obtain averaged versions.
            The average is computed as moving average with a configurable window.'''
        if name in dir(self):
            return getattrib(self, name)
        elif name in self.hist:
            return self.get_curr(name)
        elif name[:4] == 'avg_':
            if name[4:] in self.hist:
                return self.get_avg(name[4:])
            else:
                raise NotImplementedError(f'Name {name[4:]} is not available.')
        else:
            raise NotImplementedError(f'Name {name} is not available.')


class Trainer(HistoryManager):

    def __init__(self, mdl, config):
        self.mdl              = mdl
        self.n_epochs         = config['n_epochs']                     # number of epochs
        self.batch_size       = config['batch_size']                   # batch size used
        self.opt              = config['opt']                          # optimizer
        self.train_ds         = config['train_ds']                     # data loader
        self.test_ds          = config.get('test_ds')                  # the test parts are implemented in the derived class
        self.report_period    = config.get('report_period', 100)       # output a report every report_period steps
        self.n_train_samples  = config.get('n_train_samples', -1)
        if self.n_train_samples == -1:
            self.n_train_samples = sum(1 for _ in self.train_ds) * self.batch_size
            print(f'Found {self.n_train_samples} train samples.')
        self.test_period      = config.get('test_period', self.n_train_samples  // self.batch_size)
        self.n_test_samples   = config.get('n_test_samples', -1)
        if self.test_ds is not None and self.n_test_samples == -1:
            self.n_test_samples = sum(1 for _ in self.test_ds) * self.batch_size
            print(f'Found {self.n_test_samples} test samples.')
        steps_per_epoch_train = self.n_train_samples // self.batch_size
        steps_per_epoch_test  = self.n_test_samples // self.batch_size if self.test_ds is not None else None
        if isinstance(self.mdl, tf.keras.Model):
            if hasattr(self.mdl, 'in_shape'):
                self.mdl.build(input_shape=(None, *self.mdl.in_shape))        # needed for a keras summary
                self.mdl.call(layers.Input(shape=self.mdl.in_shape))
            else:
                print('Warning: no in_shape in the model specified. This is ok but the summary command may not work.')
        super().__init__(config, steps_per_epoch_train, steps_per_epoch_test)

    def train_step(self, X):
        ''' Executes one training step.
        '''
        weights = self.mdl.trainable_variables
        if len(weights) == 0:
            print('Warning, no variables found.')
        with tf.GradientTape() as tape:
            tape.watch(weights)                                     # to be on the safe side, should not be needed
            Y_pred = self.mdl(X, training=True)
            L = self.mdl.loss_fn(X=X, Y_pred=Y_pred)
        grads = tape.gradient(L, weights)
        self.opt.apply_gradients(zip(grads, weights))
        return L.numpy(), Y_pred              
    
    def train(self):
        ''' Starts the training run. It also calls tests regularly if the test functions are available
            (inherited by the TrainerWithTest class)
        '''
        self._reset_hist()
        self.overall_steps = 0
        for epoch in range(self.n_epochs):
            print(f'epoch {epoch}:')
            for step, X in enumerate(self.train_ds):
                L, Y_pred = self.train_step(X)
                m = self.metrics(L, X, Y_pred)
                for name, val in m.items():
                    self._add_to_hist(name, val)
                if step % self.report_period == 0:
                    self.train_report(epoch, step)
                if self.test_ds is not None and (step+1) % self.test_period == 0:
                    self.test(epoch, step)
                self.overall_steps += 1
        return self.hist
    
    def metrics(self, L, X, Y_pred):
        ''' Computes all the metrics which shall be reported during training. The metrics are returned
            as dictionary. May be overwritten in a derived class.
        '''
        return {'loss': L}

    def train_report(self, epoch, step):
        ''' Outputs the report. The frequency of reports can be configured. May be overwritten in a
            derived class.
        '''
        print(f"    {epoch:03d}/{step:05d}:  loss {self.loss:6.4f}")


class TrainerWithTest(Trainer): 
    
    def __init__(self, mdl, config):
        self.test_metrics = self.metrics
        super().__init__(mdl, config)

    def test_step(self, X):
        Y_pred = self.mdl(X, training=False)
        L = self.mdl.loss_fn(X=X, Y_pred=Y_pred)
        return L.numpy(), Y_pred.numpy()

    def test_report(self, epoch, step):
        print('-' * 50)
        print(f"> {epoch:03d}/{step + 1:05d}:  loss {self.test_loss:6.4f}")
        print('-' * 50)
              
    def test(self, epoch, train_step):
        '''The test is performed over all batches of the test set. We sum up all metrics.'''
        for step, X in enumerate(self.test_ds):
            L, Y_pred = self.test_step(X)
            m = self.test_metrics(L=L, X=X, Y_pred=Y_pred)
            for name, val in m.items():
                self._add_to_hist('test_'+name, val)
        for name, val in m.items():
            self._divide_hist_val_by('test_'+name, step+1)
        self.test_report(epoch, train_step)

