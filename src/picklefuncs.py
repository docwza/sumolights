import pickle

def save_data(fp, data):
    with open(fp, "wb") as fo:
        pickle.dump(data, fo)

def load_data(fp):
    with open(fp, "rb") as fo:
        data = pickle.load(fo)
    return data
