import os


if __name__ == "__main__":
    ignore = ['.DS_Store', 'dict.csv', 'rename.s']
    for id in os.listdir('../tiles'):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = id
            for tile in os.listdir('../tiles/{}'.format(dirname)):
                os.rename('../tiles/{}/{}'.format(dirname, tile), '../tiles/{}/{}_{}'.format(dirname, dirname, tile))
                