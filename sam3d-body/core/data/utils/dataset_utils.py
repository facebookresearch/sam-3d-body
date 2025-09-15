from torch.utils.data import IterableDataset, get_worker_info

class GeneratorDataset(IterableDataset):
    """
    Wrap *any* zero‑ or multi‑argument generator function as an IterableDataset.

    Args
    ----
    gen_fn : callable
        A function that returns an *iterator* (e.g. a generator).
    *args, **kwargs
        Positional / keyword arguments forwarded to `gen_fn` inside each worker.
    """
    def __init__(self, gen_fn, *args, **kwargs):
        super().__init__()
        self.gen_fn = gen_fn
        self.args   = args
        self.kwargs = kwargs

    def __iter__(self):
        """
        Called **once per DataLoader worker**. Always produce a *fresh*
        iterator here; don’t store state on `self`, or workers will collide.
        """
        worker = get_worker_info()
        if worker is None:
            # single‑process data loading
            return self.gen_fn(*self.args, **self.kwargs)
        else:
            # multi‑worker: optionally shard the stream
            it = self.gen_fn(*self.args, **self.kwargs)
            for i, sample in enumerate(it):
                # simple round‑robin sharding
                if i % worker.num_workers == worker.id:
                    yield sample
