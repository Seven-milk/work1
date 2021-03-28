# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# run theory
import numpy as np
import abc

class RuntheoryBase(abc.ABC):
    ''' Runtheory abstract class '''

    @abc.abstractmethod
    def run(self, index, threshold):
        ''' run theory function '''


class UnderRuntheory(RuntheoryBase):

    def run(self, index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray): # , axis=None
        """ Implements the RuntheoryBase.run function
        run_threshold to identify dry bell (start-end)
        point explain(discrete): start < threshold, end < threshold --> it is shrinkable and strict

        input:
            index: 1D np.ndarray, fundamental index, it should be 1d, different series identify events with different
                   length
            threshold: float, the threshold to identify dry bell(index <= threshold)

        output:
            dry_flag_start/end: 1D np.ndarray, the array contain start/end indexes of Under events
        """
        # define drought based on index and threshold
        dry_flag = np.argwhere(index <= threshold).flatten()
        self.dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
        self.dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()

        return self.dry_flag_start, self.dry_flag_end


class ExceedRuntheory(RuntheoryBase):

    def run(self, index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray):
        """ Implements the RuntheoryBase.run function
        run_threshold to identify develop period (index > threshold, different with run_threshold)
        point explain(discrete): start > threshold, end > threshold --> it is shrinkable and strict

        input:
            index: 1D np.ndarray, fundamental index
            threshold: float, the threshold to identify dry bell(index >= threshold)

        output:
            dry_flag_start/end: 1D np.ndarray, the array contain start/end indexes of Exceed events
        """
        # define develop period based on index and threshold
        dry_flag = np.argwhere(index >= threshold).flatten()
        self.dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
        self.dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()

        return self.dry_flag_start, self.dry_flag_end


if __name__ == "__main__":
    np.random.seed(15)
    index = np.random.rand(365 * 3, 2)
    threshold = 0.5
    ur = UnderRuntheory()
    er = ExceedRuntheory()
    ur_start, ur_end = ur.run(index, threshold)
    er_start, er_end = er.run(index, threshold)
