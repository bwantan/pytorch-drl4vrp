import sys
import time
from datetime import datetime

def str2bool(v):
    return v.lower() in ('true', '1')

# log the screen data to result.txt file
class Logger(object):
    def __init__(self,f=None ,stdout_print=True):
        '''
        This class is used for controlling the printing. It will write in a
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self,s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
        return time.time()

