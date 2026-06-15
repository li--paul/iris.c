import struct, zlib, sys

def analyze(fname, label):
    with open(fname, 'rb') as f:
        data = f.read()
    pos = 8
    idat = b''
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunk_data = data[pos+8:pos+8+length]
        if chunk_type == b'IDAT':
            idat += chunk_data
        pos += 12 + length
    raw = zlib.decompress(idat)
    pixels = list(raw)
    n = len(pixels)
    mean = sum(pixels) / n
    var = sum((p-mean)**2 for p in pixels) / n
    hdiffs = []
    for i in range(0, n - 3, 3):
        hdiffs.append(abs(pixels[i] - pixels[i+3]))
    hv = sum(hdiffs) / len(hdiffs) if hdiffs else 0
    print(f'{label}: mean={mean:.1f} var={var:.1f} h_adj_diff={hv:.1f} min={min(pixels)} max={max(pixels)} size={len(data)}')

analyze('test_64_4steps.png', '64x64 4steps')
analyze('test_256.png', '256x256 4steps')
analyze('test_nommap.png', '64x64 2steps')
