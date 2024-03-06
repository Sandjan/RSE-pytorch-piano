import struct
import wave

class Error(Exception):
    pass

def _read_fmt_chunk(self, chunk):
    try:
        wFormatTag, self._nchannels, self._framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from('<HHLLH', chunk.read(14))
    except struct.error:
        raise EOFError from None
    if wFormatTag == wave.WAVE_FORMAT_PCM:
        try:
            sampwidth = struct.unpack_from('<H', chunk.read(2))[0]
        except struct.error:
            raise EOFError from None
        self._sampwidth = (sampwidth + 7) // 8
        if not self._sampwidth:
            raise Error('bad sample width')
    else:
        self._sampwidth = 4
    if not self._nchannels:
        raise Error('bad # of channels')
    self._framesize = self._nchannels * self._sampwidth
    self._comptype = 'NONE'
    self._compname = 'not compressed'