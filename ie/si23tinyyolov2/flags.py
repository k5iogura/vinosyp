class FLAGs:
    _floating_infer = True

    @property
    def floating_infer(self):return FLAGs._floating_infer

    @floating_infer.setter
    def floating_infer(self, flag): FLAGs._floating_infer = flag
