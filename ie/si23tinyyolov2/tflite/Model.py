# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class Model(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsModel(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Model()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ModelBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Model
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Model
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Model
    def OperatorCodes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .OperatorCode import OperatorCode
            obj = OperatorCode()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def OperatorCodesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def Subgraphs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .SubGraph import SubGraph
            obj = SubGraph()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def SubgraphsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def Description(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Model
    def Buffers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Buffer import Buffer
            obj = Buffer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def BuffersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def ModelStart(builder): builder.StartObject(5)
def ModelAddVersion(builder, version): builder.PrependUint32Slot(0, version, 0)
def ModelAddOperatorCodes(builder, operatorCodes): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(operatorCodes), 0)
def ModelStartOperatorCodesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ModelAddSubgraphs(builder, subgraphs): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(subgraphs), 0)
def ModelStartSubgraphsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ModelAddDescription(builder, description): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(description), 0)
def ModelAddBuffers(builder, buffers): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(buffers), 0)
def ModelStartBuffersVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ModelEnd(builder): return builder.EndObject()
