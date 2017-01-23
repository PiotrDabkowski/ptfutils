from Queue import Queue
from threading import Thread
import random
import time
import os


class BatchManager:
    def __init__(self, example_getter, example_indices, batch_composer, examples_per_batch, shuffle_examples=True, num_workers=6):
        ''' example_getter is a function taking a list of example indices and returning a list of examples (with labels)
                      whatever format you want, but one full example per element
            batch_composer is another function which takes examples_per_batch items returned by example_getter and composes them
                           into a full batch, again whatever format you want '''
        self.example_indices = tuple(example_indices)
        self.example_getter = example_getter
        self.batch_composer = batch_composer
        self.examples_per_batch = examples_per_batch

        self.shuffle_examples = shuffle_examples
        self.num_workers = num_workers
        if self.shuffle_examples:
            self.reshufle_examples()
        # make number of examples divisible by batch size - makes everything simpler and does not change the results significantly
        self.example_indices += self.example_indices[: (-len(self.example_indices)) % self.examples_per_batch]
        assert len(self.example_indices) % self.examples_per_batch == 0


        self.total_examples = len(self.example_indices)
        self.total_batches = self.total_examples / self.examples_per_batch

        self.request_queue = Queue()

        self.ready_queue = Queue()

        self.target_ready_size = 3 * self.num_workers
        assert 1.2*self.target_ready_size < self.total_batches

        self.run_workers()

        self.iterating = False
        self.current_index = None
        self.last_requested = None
        self._allow_one_time_continue = False


    def get_state(self):
        return {
            'iterating': self.iterating,
            'current_index': self.current_index,
            'last_requested': self.last_requested,
            'example_indices': self.example_indices
        }

    def continue_from_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        if state['iterating']:
            while not self.request_queue.empty():
                self.request_queue.get()
            time.sleep(3)
            while not self.ready_queue.empty():
                self.ready_queue.get()
            self._allow_one_time_continue = True
            last_requested = self.last_requested
            self.last_requested = self.current_index
            for e in xrange(self.current_index, last_requested):
                self.request_more()
            assert self.last_requested == last_requested

    def run_workers(self):
        for w in xrange(self.num_workers):
            t = Thread(target=self._worker)
            t.daemon = True
            t.start()


    def reshufle_examples(self):
        ex = list(self.example_indices)
        random.shuffle(ex)
        self.example_indices = tuple(ex)


    def __iter__(self):
        if self._allow_one_time_continue:
            self._allow_one_time_continue = False
            return self
        if self.iterating:
            raise RuntimeError('Already iterating!')

        # make sure both queues are empty
        assert self.request_queue.empty()
        assert self.ready_queue.empty()

        if self.shuffle_examples:
            self.reshufle_examples()

        self.iterating = True
        self.current_index = 0
        self.last_requested = 0
        # fill request queue with target amount of requests
        for e in xrange(self.target_ready_size):
            self.request_more()
        return self

    def request_more(self):
        start = self.last_requested * self.examples_per_batch
        if start >= len(self.example_indices):
            return False # no more data to request, undo and return

        # put a tuple containing examples for the next batch
        self.request_queue.put(self.example_indices[start:start+self.examples_per_batch])
        self.last_requested += 1
        return True

    def next(self):
        if self.current_index >= self.total_batches:
            self.iterating = False
            self.current_index = None
            self.last_requested = None
            raise StopIteration('Epoch ended')
        res = self.ready_queue.get(timeout=60)  # <- terrible, terrible error
        self.current_index += 1
        self.request_more()
        return res

    def _worker(self):
        while True:
            task = self.request_queue.get()
            self.ready_queue.put(self.batch_composer(self.example_getter(task)))


if __name__=='__main__':
    a = Queue()
    bm = BatchManager(lambda x: list(x), range(12800), lambda x: tuple(x), 3, True, num_workers=6)
    x = []
    t = time.time()
    for e in bm:
        x.append(e)
    assert len(x)==len(set(x))==bm.total_batches
    print 'Overhead per batch -> ', (time.time() - t) / bm.total_batches
    y = []
    for e in bm:
        y.append(e)
    assert len(y) == len(set(y)) == bm.total_batches
    assert len(set(x+y)) > 1.9*bm.total_batches


    # test state continue
    print 'T2'
    bm = BatchManager(lambda x: list(x), range(12800), lambda x: tuple(x), 1, True, num_workers=6)

    z = []
    lim = bm.total_batches
    i = 0
    for e in bm:
        i += 1
        z.append(e)
        if i > lim/2:
            state = bm.get_state()
            break
    print 'co'
    bm.continue_from_state(state)
    print 'er'
    for e in bm:
        z.append(e)
    i = 0
    print 'se'
    for e in bm:
        i+=1
    assert i==lim==len(z)==len(set(z))




