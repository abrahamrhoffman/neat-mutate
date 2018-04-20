from __future__ import print_function
from reprint import output
import time, threading, random

class Report(object):
    '''
    Mutate StdOut Reporting
    '''

    def __init__(self, data, FITNESS):
        self.fitness = FITNESS
        self.data = data

    def start(self):
        X,Y = self.data
        result,error,solved = self.fitness

        print('Expected:')
        print('{}'.format(Y))
        print('Result:')
        print('{}'.format(result))
        print('Error:')
        print('{}'.format(error))


    def start_cool(self, index=0):
        LENGTH = 100
        global output_list
        now = 0
        while now < LENGTH:
            max_step = LENGTH-now
            step = random.randint(1, max_step or 1)
            now += step

        output_list[index] = "{}{}{padding}{ending}".format(
            "-" * now,
            index,
            padding=" " * (LENGTH-now-1) if now < LENGTH else "",
            ending="|" if now < LENGTH else ""
        )

        time.sleep(1)

        output_list.append("{} finished".format(index))

        with output(output_type="list", initial_len=5, interval=0) as output_list:
            pool = []
            for i in range(5):
                t = threading.Thread(target=some_op, args=(i,))
                t.start()
                pool.append(t)
            [t.join() for t in pool]

            with output(initial_len=3, interval=0) as output_lines:
                while True:
                    output_lines[0] = "First_line  {}...".format(random.randint(1,10))
                    output_lines[1] = "Second_line {}...".format(random.randint(1,10))
                    output_lines[2] = "Third_line  {}...".format(random.randint(1,10))
                    time.sleep(0.5)
