from itertools import islice


def batch_generator(data_generator, batch_size):
    while True:
        batch = list(islice(data_generator, batch_size))
        if not batch:
            break
        yield batch
