import struct

with open('test_user.png', 'rb') as f:
    data = f.read()

print(f'Size: {len(data)}')
sig = data[:8]
print(f'Signature: {sig.hex()}')

pos = 8
while pos < len(data):
    length = struct.unpack('>I', data[pos:pos+4])[0]
    chunk_type = data[pos+4:pos+8]
    type_str = chunk_type.decode('ascii', errors='replace')
    print(f'  Chunk: {type_str} length={length}')
    if chunk_type == b'IHDR':
        w = struct.unpack('>I', data[pos+8:pos+12])[0]
        h = struct.unpack('>I', data[pos+12:pos+16])[0]
        bd = data[pos+16]
        ct = data[pos+17]
        print(f'    Width={w} Height={h} bit_depth={bd} color_type={ct}')
    if chunk_type == b'IDAT':
        print(f'    IDAT starts: {data[pos+8:pos+16].hex()}')
    pos += 12 + length
