# coding=utf-8
import numpy as np
from configs.hyperparams import TrainConfig


def list2array(_list):
    return np.array(list(map(float, _list)))


def array2string(array):
    return ' '.join(map(str, array.tolist()))


class EmbdMapper:
    def __init__(self, data_path, config):
        self.data_path = data_path
        # special char
        self.eos_char = config.eos_char
        self.go_char = config.go_char
        self.unk_char = config.unk_char
        self.pad_char = config.pad_char
        # load embedding
        self.num, self.dim, self.embd_dict = self.load_embd()

    def load_embd(self):
        embd_dict = dict()
        with open(self.data_path, "r") as f:
            # read meta data
            meta_line = f.readline()
            _, dim = meta_line.split()
            dim = int(dim)
            while 1:
                lines = f.readlines(1000)
                if not lines:
                    break
                for line in lines:
                    chars, *scalars = line[:-1].split()
                    if len(chars) > 1:
                        continue
                    # extend 3 dimension for embedding another 4 special chars.
                    scalars.extend([0, 0, 0])
                    array = list2array(scalars)
                    print(sum(array))
                    # print(np.sum(np.square(array)))
                    embd_dict[chars] = array
        # add 4 special chars
        zeros = np.zeros(shape=(1, dim))
        print(zeros.shape)
        embd_dict[self.go_char] = np.concatenate((zeros, [[1, 0, 0]]), axis=1)
        embd_dict[self.eos_char] = np.concatenate((zeros, [[0, 1, 0]]), axis=1)
        embd_dict[self.unk_char] = np.concatenate((zeros, [[0, 0, 1]]), axis=1)
        embd_dict[self.pad_char] = np.concatenate((zeros, [[0, 0, 0]]), axis=1)
        num = len(embd_dict.keys())
        print('number of vector:{}, dimension:{}'.format(num, dim+3))
        return num, dim+3, embd_dict

    def map(self, char):
        if char not in self.embd_dict.keys():
            raise KeyError
        return self.embd_dict[char]

    def get_char2idx(self):
        char2idx = {char: idx for idx, char in enumerate(self.embd_dict.keys())}
        return char2idx

    def get_lookup_table(self):
        return list2array(self.embd_dict.values())

    def save(self, path):
        embd_list = list()
        for k in self.embd_dict.keys():
            embd_list.append('{} {}\n'.format(k, array2string(self.embd_dict[k])))
        sorted(embd_list)
        with open(path, 'w', newline='\n') as f:
            f.writelines(['{} {}\n'.format(self.num, self.dim)])
            f.writelines(embd_list)


def main():
    path = 'data/embd/sgns.renmin.char'
    config = TrainConfig()
    em = EmbdMapper(path, config)
    em.save('data/embd/sgns.renmin.char.reduce')


if __name__ == '__main__':
    main()
