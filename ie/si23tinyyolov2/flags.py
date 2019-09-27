global __floating_infer
class FLAGs:
    def __init__(initval=True):
        global __floating_infer
        __floating_infer = True

    @property
    def floating_infer(self):
        global __floating_infer
        return(__floating_infer)

    @floating_infer.setter
    def floating_infer(self,val):
        global __floating_infer
        __floating_infer = val
flags = FLAGs()
