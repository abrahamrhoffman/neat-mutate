from __future__ import print_function
from multiprocessing import Process, Queue
import sys, os, time, subprocess
from functools import partial
from report import spinner, bar

class Progress(object):
    '''
    Mutate StdOut Reporting
    '''

    def __init__(self):
        pass
        #self.fitness = FITNESS
        #self.data = data

    def start(self):
        queue = Queue()
        p = Process(target=self.statistics, args=(queue))
        p.start()
        p.join()
        result = queue.get()
        print(result)
        self.statistics()

    def statistics(self):
        s = spinner.PixelSpinner('Evolving Models: ')
        i = 0
        while i <= 100:
            time.sleep(0.1)
            i += 1
            s.next()

        print()

        #b = bar.Bar('Loading', fill='@', suffix='%(percent)d%%')
        #for i in range(100):
        #    time.sleep(0.1)
        #    b.next()
        #b.finish()


        #X,Y = self.data
        #result,error,solved = self.fitness

        #print('Expected:')
        #print('{}'.format(Y))
        #print('Result:')
        #print('{}'.format(result))
        #print('Error:')
        #print('{}'.format(error))
