# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# work flow class, 未来需要进一步改进，如workflow可以写成容器，设置set/get方法、workbase基类设置抽象属性等
import abc


class WorkBase(abc.ABC):
    ''' WorkBase abstract class '''

    @abc.abstractmethod
    def job(self):
        ''' define job for this work '''


class WorkFlow:
    ''' WorkFlow, work manager: add work and run '''

    def __init__(self, *args: WorkBase):
        ''' init function
        self.work: a list contains subclasses of WorkBase to manager work
        '''
        self.work = [work for work in args]

    def addwork(self, work: WorkBase):
        ''' add work
        input:
            draw: DrawBase class
        '''
        self.work.append(work)

    def addworklist(self, worklist: list):
        ''' add work list
        input:
            worklist: list contains subclasses of WorkBase
        '''
        self.work.extend(worklist)

    def delwork(self, index=-1):
        ''' del work
        input:
            index: the index of work which will be deleted, default -1 (last work)
        '''
        del self.work[index]

    def showworks(self):
        ''' show works '''
        for i in range(len(self.work)):
            print(f"work{i}: ", self.work[i], '\n')

    def runflow(self):
        ''' run workflow '''
        for i in range(len(self.work)):
            print(f"running the work{i}: ", self.work[i], "...\n")
            self.work[i].job()
            print(f"work{i} complete\n")
