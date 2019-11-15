from PushInput_io import PushInput_io
import ipdb 
st = ipdb.set_trace
if __name__ == '__main__':
    input_collection_to_number = {'train': 0, 'val': 1, 'test': 2}
    inputio = PushInput_io()
    sample = inputio.data(0)
    # st()

