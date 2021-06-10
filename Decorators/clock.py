# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import time
import functools


def clock(func):
    ''' decorator, print the time elapsed (and results) for func running '''
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(','.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(','.join(pairs))
        arg_str = ','.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked


@clock
def test1():
    for i in range(5):
        print(i)


if __name__ == '__main__':
    test1()