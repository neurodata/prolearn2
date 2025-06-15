import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataclasses import dataclass


class DataGeneratingProcess:
    """data generating process"""
    def __init__(self, period, seq_len, num_seeds, 𝜆=5.0, p=0.0):
        self.period = period
        self.seq_len = seq_len
        self.num_seeds = num_seeds
        self.𝜆 = 𝜆 # poisson rate (number of samples received within a unit time interval)
        self.p = p # sparse rate (probability of not receiving any samples at a given time)

    def generate_data(self):
        """generate data sequences over the specified number of seeds"""
        xseq, yseq, tseq, taskseq = [], [], [], []
        for _ in range(self.num_seeds):
            dat = self.generate_sequence(np.random.randint(0, 10000))
            xseq.append(dat[0])
            yseq.append(dat[1])
            tseq.append(dat[2])
            taskseq.append(dat[3])

        self.data = {'x': xseq,
                     'y': yseq,
                     't': tseq,
                     'task': taskseq}

    def generate_sequence(self, seed):
        """generate a sequence of data"""
        np.random.seed(seed)

        T = self.period

        n_list = np.random.poisson(self.𝜆, self.seq_len) if self.𝜆 > 0 else np.ones(self.seq_len, dtype=int)
        drop_list = np.random.binomial(1, self.p, self.seq_len).astype(bool)

        data = []
        taskseq = []
        for t, (n, drop) in enumerate(zip(n_list, drop_list)):
            current_task = (t % T) < (T // 2)
            taskseq.append(int(current_task))

            if drop:
                continue

            x1 = np.random.uniform(-2, -1, n)
            x2 = np.random.uniform(1, 2, n)
            mask = np.random.choice([0, 1], p=[0.5, 0.5], size=n)
            x = x1 * mask + x2 * (1 - mask)

            if current_task:
                y = x > 0
            else:
                y = x < 0      
        
            data.append([x, y, t * np.ones(n)])
        
        data = np.concatenate(data, axis=-1)
        Xdat = data[0, :].reshape(-1, 1)
        Ydat = data[1, :].astype(int)
        tind = data[2, :]
        return Xdat, Ydat, tind, taskseq
    
    def generate_at_time(self, t, num_samples):
        """generate a test sample of data (x, y) ~ p_t from the marginal 
        of the process at time t. This is used to evaluate the instantaneous
        loss of the predictor

        might not be needed
        """
        # Generate samples from U[-2, -1] union U[1, 2]
        x1 = np.random.uniform(-2, -1, num_samples)
        x2 = np.random.uniform(1, 2, num_samples)
        mask = np.random.choice([0, 1], p=[0.5, 0.5], size=num_samples)
        Xdat = x1 * mask + x2 * (1 - mask)

        # create labels
        T = self.period
        if (t % T) < (T // 2):
            Ydat = Xdat > 0
        else:
            Ydat = Xdat < 0

        Xdat = Xdat.reshape(-1, 1)
        tdat = t * np.ones(num_samples)

        x = torch.from_numpy(Xdat).float()
        y = torch.from_numpy(Ydat).long()
        t = torch.from_numpy(tdat).float()
        return x, y, t
    

class PolynomialProcess:
    """data generating process"""
    def __init__(self, type='linear', seq_len=2000, num_seeds=3, num_samples_per_task=100):
        self.type = type
        self.seq_len = seq_len
        self.num_seeds = num_seeds
        self.n = num_samples_per_task
        
    def generate_data(self):
        """generate data sequences over the specified number of seeds"""
        xseq, yseq, taskseq = [], [], []
        tseq = []
        for _ in range(self.num_seeds):
            dat = self.generate_sequence(np.random.randint(0, 1000))
            xseq.append(dat[0])
            yseq.append(dat[1])
            taskseq.append(dat[2])
            tseq.append(dat[2])
            
        xseq = np.array(xseq)
        yseq = np.array(yseq)
        tseq = np.array(tseq)
        taskseq = np.array(taskseq)

        self.data = {'x': xseq,
                     'y': yseq,
                     't': tseq,
                     'task': taskseq}

    def generate_sequence(self, seed):
        """generate a sequence of data"""
        np.random.seed(seed)

        data = []
        for s in range(self.seq_len):
            x, y, t = self.generate_at_time(s, self.n)
            data.append([x, y, t])
        data = np.concatenate(data, axis=-1)
        Xdat = data[0, :].reshape(-1, 1)
        Ydat = data[1, :].astype(int)
        tind = data[2, :]
        return Xdat, Ydat, tind
    
    def generate_at_time(self, t, num_samples, return_tensors=False):
        """generate a test sample of data (x, y) ~ p_t from the marginal 
        of the process at time t. This is used to evaluate the instantaneous
        loss of the predictor
        """
        m = 0.1
        if self.type == 'linear':
            trend = lambda t : m * t
        elif self.type == 'quadratic':
            trend = lambda t : m * t**2
        else:
            raise ValueError(f"Unknown process type: {self.type}")

        x1 = np.random.uniform(trend(t)-11, trend(t)-10, num_samples // 2)
        x2 = np.random.uniform(trend(t)+10, trend(t)+11, num_samples // 2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((np.ones(num_samples // 2), np.zeros(num_samples // 2)))

        ts = t * np.ones(num_samples)

        if return_tensors:  
            x = torch.from_numpy(x.reshape(-1, 1)).float()
            y = torch.from_numpy(y.reshape(-1)).long()
            t = torch.from_numpy(ts.reshape(-1)).float()
            return x, y, t
        else:
            return x, y, ts
        

class ABABProcess:
    """data generating process"""
    def __init__(self, period=20, seq_len=10000, num_seeds=3, num_samples_per_task=20):
        self.seq_len = seq_len
        self.num_seeds = num_seeds
        self.period = period
        self.num_samples_per_task = num_samples_per_task

    def generate_data(self):
        """generate data sequences over the specified number of seeds"""
        xseq, yseq, tseq = [], [], []
        for _ in range(self.num_seeds):
            dat = self.generate_sequence(np.random.randint(0, 10000))
            xseq.append(dat[0])
            yseq.append(dat[1])
            tseq.append(dat[2])

        self.data = {
            'x': xseq,
            'y': yseq,
            't': tseq,
        }

    def generate_sequence(self, seed):
        """generate a sequence of data"""
        np.random.seed(seed)

        data = []
        for s in range(self.seq_len):
            x, y, t = self.generate_at_time(s, self.num_samples_per_task)
            data.append([x, y, t])
        data = np.concatenate(data, axis=-1)
        Xdat = data[0, :].reshape(-1, 1)
        Ydat = data[1, :].astype(int)
        tind = data[2, :]
        return Xdat, Ydat, tind

    
    def generate_at_time(self, t, num_samples, return_tensors=False):
        """generate a test sample of data (x, y) ~ p_t from the marginal 
        of the process at time t. This is used to evaluate the instantaneous
        loss of the predictor

        might not be needed
        """
        # Generate samples from U[-2, -1] union U[1, 2]
        x1 = np.random.uniform(-2, -1, num_samples)
        x2 = np.random.uniform(1, 2, num_samples)
        mask = np.random.choice([0, 1], p=[0.5, 0.5], size=num_samples)
        Xdat = x1 * mask + x2 * (1 - mask)

        # create labels
        T = self.period
        if (t % T) < (T // 2):
            Ydat = Xdat > 0
        else:
            Ydat = Xdat < 0

        tdat = t * np.ones(num_samples)

        if return_tensors:
            x = torch.from_numpy(Xdat.reshape(-1, 1)).float()
            y = torch.from_numpy(Ydat.reshape(-1)).long()
            t = torch.from_numpy(tdat.reshape(-1)).float()
            return x, y, t
        else:
            return Xdat, Ydat, tdat
    

class SyntheticDataset(Dataset):
    """Form the torch dataset"""
    def __init__(self, data, present, run_id, test, past=None):
        self.x = torch.from_numpy(data['x'][run_id]).float()
        self.y = torch.from_numpy(data['y'][run_id]).long()
        self.t = torch.from_numpy(data['t'][run_id]).float()

        if test:
            # Use data from time 'present' onwards for testing
            test_idx = torch.where(self.t >= present)
            self.x = self.x[test_idx]
            self.y = self.y[test_idx]
            self.t = self.t[test_idx]
        else:
            if past is None:
                #  # Use data up to time 'idx' onwards for training (full history)
                train_idx = torch.where(self.t < present)
                self.x = self.x[train_idx]
                self.y = self.y[train_idx]
                self.t = self.t[train_idx]
            else:
                # Use the most recent past data up to time 'idx' onwards for training (partial history)
                train_idx = torch.where((self.t >= present-past) & (self.t < present))
                self.x = self.x[train_idx]
                self.y = self.y[train_idx]
                self.t = self.t[train_idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        return x, y, t
    

class NormalizedSyntheticDataset(Dataset):
    """Form the torch dataset"""
    def __init__(self, data, present, run_id, test, normalize_x=False, normalize_t=False):
        x = torch.from_numpy(data['x'][run_id]).float()
        y = torch.from_numpy(data['y'][run_id]).long()
        t = torch.from_numpy(data['t'][run_id]).float()

        train_idx = torch.where(t < present)
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        t_tr = t[train_idx]

        test_idx = torch.where(t >= present)
        x_te = x[test_idx]
        y_te = y[test_idx]
        t_te = t[test_idx]

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.t_tr = t_tr
        
        if test:
            self.x = self.minimax_normalize(x_te, x_tr) if normalize_x else x_te
            self.t = self.minimax_normalize(t_te, t_tr) if normalize_t else t_te
            self.y = y_te
        else:
            self.x = self.minimax_normalize(x_tr, x_tr) if normalize_x else x_tr
            self.t = self.minimax_normalize(t_tr, t_tr) if normalize_t else t_tr
            self.y = y_tr

    def minimax_normalize(self, a, a_tr):
        tr_min, tr_max = a_tr.min(), a_tr.max()
        return (a - tr_min)/(tr_max - tr_min)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        return x, y, t