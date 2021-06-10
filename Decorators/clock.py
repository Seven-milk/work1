# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import time


def clock(func):
    ''' decorator, print the time elapsed (and results) for func running '''
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_str = ','.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked


@clock
def test1():
    for i in range(5):
        print(i)


if __name__ == '__main__':
    test1()