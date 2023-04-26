import os

class TrainLogger:
    """
    Logger for DNN-based scripts
    
    Initialized with out path,
    could write to file or/and print to console
    """
    def __init__(
        self, 
        outpath,
        ctx='cpu'
    ):
        self.out = outpath
        self.ctx = ctx
        if os.path.isfile(self.out):
            with open(self.out, 'w') as f:
                pass
        else:
            if not os.path.isdir(os.path.split(self.out)[0]):
                os.makedirs(os.path.split(self.out)[0])
            with open(self.out, 'w') as f:
                pass
        print('log file is opened at', self.out)
    
    def write(self, m):
        try:
            with open(self.out, 'a') as f:
                f.write(m)
        except:
            pass
    
    def print(self, m, show=True, write=True):
        # m = self.ctx + '\n' + m
        if show:
            print(m)
        if write:
            self.write(m+'\n')

    def writeAsCSV(self, header, metrics):
        def processRow(row):
            info = ''
            for i, m in enumerate(row):
                if isinstance(m, float):
                    info += '{:6f}'.format(m)
                else:
                    info += '{}'.format(m)
                if i == len(row)-1:
                    info += '\n'.format(m)
                else:
                    info += ','.format(m)
            return info
        self.write(processRow(header))
        for metric in metrics:
            self.write(processRow(metric))
            
        
