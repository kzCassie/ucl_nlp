## Class struct to store parameters
from common.utils import init_arg_parser

class Parameters():
    def __init__(self):
        arg_parser = init_arg_parser()
        print(arg_parser)

train_param = Parameters()
