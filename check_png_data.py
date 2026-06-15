import struct
import zlib

with open('test_nommap.png', 'rb') as f:
    data = f.read()

pos = 8
width = height = 0
idat_data = b''

while pos < len(data):
    length = struct.unpack('>I', data[pos:pos+4])[0]
    chunk_type = data[pos+4:pos+8]
    chunk_data = data[pos+8:pos+8+length]
    if chunk_type == b'IHDR':
        width = struct.unpack('>I', chunk_data[0:4])[0]
        height = struct.unpack('>I', chunk_data[4:8])[0]
        bit_depth = chunk_data[8]
        color_type = chunk_data[9]
    elif chunk_type == b'IDAT':
        idat_data += chunk_data
    pos += 12 + length

# Decompress IDAT
raw = zlib.decompress(idat_data)
print(f'Raw size: {len(raw)} bytes')
print(f'Expected: {width}*{height}*3 = {width*height*3}')

# Check pixel values
pixels = list(raw)
print(f'Min pixel: {min(pixels)}')
print(f'Max pixel: {max(pixels)}')
print(f'Mean pixel: {sum(pixels)/len(pixels):.1f}')

# Check if all same
if len(set(pixels)) == 1:
    print('WARNING: All pixels are the same color!')
else:
    print(f'Unique pixel values: {len(set(pixels))}')
    print(f'First 10 pixels: {pixels[:10]}')
    # Check if it looks like noise or an image
    hist = [0]*256
    for p in pixels:
        hist[p] += 1
    nonzero = sum(1 for h in hist if h > 0)
    print(f'Histogram buckets used: {nonzero}/256')
