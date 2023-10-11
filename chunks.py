import os
def split_file(input_path, output_path, chunk_size=10 * 1024 * 1024):
        with open(input_path, 'rb') as file:
         chunk_num = 0
         while True:
            data = file.read(chunk_size)
            if not data:
                break
            chunk_num += 1
            output_file = os.path.join(output_path, f'chunk_{chunk_num:03d}.bin')
            with open(output_file, 'wb') as chunk:
                chunk.write(data)

ip=input('Enter Input Path: ')
op=input('Enter Output Path: ')
split_file(ip, op)
