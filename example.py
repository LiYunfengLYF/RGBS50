import multiprocessing

from rgbs50 import ExperimentRGBS
from rgbs50.run import TrackerFactory

if __name__ == '__main__':

    data_root = r''
    project_root = r''
    multiprocessing.set_start_method('spawn')
    e = ExperimentRGBS(data_root, project_root)
    tracker_params = [
        # {'name': 'SiamRPN'},
        # {'name': 'SiamRPNpp', 'version': 'siamrpnpp_resnet'},
        # {'name': 'SiamRPNpp', 'version': 'siamrpnpp_mobilenet'},
        # {'name': 'SiamCAR'},
        # {'name': 'SiamBAN'},
        # {'name': 'SiamBAN_ACM'},
        # {'name': 'OSTrack', 'version': 'ostrack256'},
        # {'name': 'OSTrack', 'version': 'ostrack384'},
        # {'name': 'Stark', 'version': 'stark_s50'},
        # {'name': 'Stark', 'version': 'stark_st50'},
        # {'name': 'Stark', 'version': 'stark_st101'},
        # {'name': 'TransT'},
        # {'name': 'TransT_SLT'},
    ]

    tracker_factory = TrackerFactory()

    for params in tracker_params:
        e.run(tracker_factory(params))

        e.multi_run(params, tracker_factory, threads=4)
        e.eval(tracker_factory(params).name)
#
