"""Output utilities"""

import os
import sys
import contextlib
from inspect import currentframe, getsourcefile
from typing import Callable, Union


@contextlib.contextmanager
def redirect_output(stdout_function: Union[Callable, None] = None):
    """Redirect output"""
    if os.name == 'nt':
        yield
    else:
        stdout_fileno = sys.stdout.fileno()
        stdout_original = os.dup(stdout_fileno)
        stdout_pipe_read, stdout_pipe_write = os.pipe()
        os.dup2(stdout_pipe_write, stdout_fileno)
        os.close(stdout_pipe_write)
        try:
            yield
        finally:
            os.close(stdout_fileno)
            if stdout_function is not None:
                captured_stdout = ''
                while True:
                    data = os.read(stdout_pipe_read, 256)
                    if not data:
                        break
                    captured_stdout += data.decode('utf-8')
            os.close(stdout_pipe_read)
            os.dup2(stdout_original, stdout_fileno)
            os.close(stdout_original)
            if stdout_function is not None:
                frame = currentframe().f_back.f_back
                header = '[REDIRECT] {}::{}::{}():\n>   '.format(
                    os.path.basename(getsourcefile(frame)),
                    frame.f_code.co_name,
                    frame.f_lineno,
                )
                captured_stdout = header + captured_stdout.replace(
                    '\n',
                    '\n>   ',
                )
                stdout_function(captured_stdout)
