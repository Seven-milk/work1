# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# work flow class, 未来需要进一步改进，如workflow可以写成容器，设置set/get方法、workbase基类设置抽象属性等
import abc
from importlib import reload


class WorkBase(abc.ABC):
    ''' WorkBase abstract class '''

    @abc.abstractmethod
    def run(self):
        ''' define something to run for this work '''


class WorkExample(WorkBase):
    ''' Work, a example class '''

    def __init__(self, info: str = "", *args, **kwargs):
        ''' init function '''
        self._info = info
        self._args = args
        self._kwargs = kwargs

    def run(self):
        ''' Implement the WorkBase.run '''
        print(f"I'm running: {self._info}{self._args}{self._kwargs}...I have completed this work")
        return None

    def __repr__(self):
        return f"This is WorkExample, info: {self._info}"

    def __str__(self):
        return f"This is WorkExample, info: {self._info}"


class WorkFlow:
    ''' WorkFlow, work manager: add work and run '''

    def __init__(self, *args: WorkBase):
        ''' init function
        input:
            *args: position args, WorkFlow(WorkBase1, WorkBase2...)
        self._works: a list contains subclasses of WorkBase to manager work
        '''
        self._works = [work for work in args]
        self.ret = []

    def __getitem__(self, item):
        return self._works[item]

    def __len__(self):
        return len(self._works)

    def __str__(self):
        str_ = "WorkFlow:\n" + "\n".join([f"work{i}: {str(self._works[i])}" for i in range(len(self._works))])
        return str_

    def __repr__(self):
        str_ = "WorkFlow:\n" + "\n".join([f"work{i}: {str(self._works[i])}" for i in range(len(self._works))])
        return str_

    def __add__(self, other):
        if isinstance(other, WorkFlow):
            return WorkFlow(*(self._works + other._works))
        elif isinstance(other, WorkBase):
            return WorkFlow(*(self._works + [other]))
        else:
            raise TypeError("Input should be a instance of WorkFlow or WorkBase")

    def __iadd__(self, other):
        if isinstance(other, WorkFlow):
            self._works.extend(other._works)
        elif isinstance(other, WorkBase):
            self._works.append(other)
        else:
            raise TypeError("Input should be a instance of WorkFlow or WorkBase")
        return self

    def __delitem__(self, key):
        del self._works[key]
        return None

    def __setitem__(self, key, value: WorkBase):
        if isinstance(value, WorkBase):
            self._works[key] = value
        else:
            raise TypeError("Input should be a instance of WorkBase")
        return None

    def runflow(self, key: list = None, skip: list = None):
        ''' run WorkFlow: run works in this flow
        input:
            key: list, work to do
            skip: list, work to skip
        '''
        # default set
        if key == None:
            key = list(range(len(self._works)))
        if skip == None:
            skip = []

        print(f"Start running WorkFlow\n")
        i = 0
        for work in self._works:
            if i not in key or i in skip:
                i += 1
                continue
            print(f"Start running work{i}: {str(work)}")
            # append work.run() results in self._ret
            ret_ = work.run()
            self.ret.append({f"work{i}_ret": ret_})
            i += 1
            print(f"Complete wrok{i}: {str(work)}\n")
        print(f"\nComplete WorkFlow")


if __name__ == '__main__':
    we1 = WorkExample("work0", 1, 2, people=1)
    we2 = WorkExample("work1", people=2)
    print(we1)
    we1.run()
    wf = WorkFlow(we1)
    print(wf)
    wf += we2
    print(wf)
    wf += wf
    print(wf)
    del wf[0]
    print(wf)
    wf.runflow()
