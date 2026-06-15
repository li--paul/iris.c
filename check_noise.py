import struct, zlib

for fname in ['test_nommap.png', 'test_st.png']:
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
    mean = sum(pixels)/n
    var = sum((p-mean)**2 for p in pixels)/n
    # Check spatial correlation: diff between adjacent pixels
    hdiff = sum(abs(pixels[i] - pixels[i+1]) for i in range(0, len(pixels)-1, 3)) / (len(pixels)//3)
    print(f'{fname}: mean={mean:.1f} var={var:.1f} avg_adj_diff={hdiff:.1f} min={min(pixels)} max={max(pixels)}')
