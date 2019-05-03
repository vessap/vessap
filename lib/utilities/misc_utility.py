import numpy
try:
    import cPickle as pickle
except:
    import pickle


def save_data(filename,data):
    s = open(filename,'wb')
    try:
        pickle.dump(data,s)
    except:
        print 'unable to save data to file %s'%(filename,)

def load_data(filename):
    s = open(filename,'rb')
    try:
        data = pickle.load(s)
        return data
    except:
        print 'unable to load data from file %s'%(filename,)

    s.close()


def shuffle_in_unison(array_list):
    if type(array_list)==list  or type(array_list) ==tuple:
        first_item = array_list[0]
        for arr in array_list:
            assert len(first_item) == len(arr)

        output = []
        for arr in array_list:
            output.append(numpy.empty(arr.shape, dtype=arr.dtype))

        permutation = numpy.random.permutation(len(first_item))

        for old_index, new_index in enumerate(permutation):
            for i in range(len(output)):
                output[i][new_index] = array_list[i][old_index]

        return output
    else:
        return ()
