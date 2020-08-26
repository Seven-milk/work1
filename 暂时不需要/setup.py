# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from distutils.core import setup

setup(
    name='EDI_refactoring',
    version='v1.0',
    author='ZhengXudong',
    author_email='Z786909151@163.com',
    py_modules=['EDI_refactoring.ac_P','EDI_refactoring.create_Dataframe','EDI_refactoring.data_write','EDI_refactoring.data_read',
               'EDI_refactoring.DS_cal','EDI_refactoring.DS_modify','EDI_refactoring.EDI','EDI_refactoring.fun_EP',
               'EDI_refactoring.fun_PRN','EDI_refactoring.move_average','EDI_refactoring.plot_EDI',
               'EDI_refactoring.run_theory','EDI_refactoring.stastic'],
    data_files='EDI_refactoring.P_test',
    requires=['numpy', 'math', 'pandas','matplotlib', 'datetime'],
)

