'''
Rename Multiple Files In a Directory
'''

import os

def main():
    i = 1

    for filename in os.listdir('./orange'):
        dst = 'orange' + '_' + str(i) + '.jpg'
        src = filename

        os.rename(os.path.join('./orange', src), os.path.join('./orange', dst))
        
        i += 1
        
if __name__ == '__main__':
    main()

