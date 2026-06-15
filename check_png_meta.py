import struct

with open('test_nommap.png', 'rb') as f:
    data = f.read()

pos = 8
while pos < len(data):
    length = struct.unpack('>I', data[pos:pos+4])[0]
    chunk_type = data[pos+4:pos+8]
    chunk_data = data[pos+8:pos+8+length]
    type_str = chunk_type.decode('ascii', errors='replace')
    if chunk_type == b'tEXt':
        null_pos = chunk_data.find(b'\x00')
        key = chunk_data[:null_pos].decode('ascii', errors='replace')
        value = chunk_data[null_pos+1:].decode('ascii', errors='replace')
        print(f'tEXt: {key} = {value}')
    elif chunk_type == b'IDAT':
        print(f'IDAT: {length} bytes of compressed pixel data')
        break
    pos += 12 + length
