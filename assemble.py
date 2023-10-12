import os
def join_chunks(chunk_dir, output_file):
    import os
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.bin')]
    chunk_files.sort()
    with open(output_file, 'wb') as outfile:
        for chunk_file in chunk_files:
            with open(os.path.join(chunk_dir, chunk_file), 'rb') as chunk:
                data = chunk.read()
                outfile.write(data)

ip=input('Input Directory: ')
op=input('Output File Name: ')
