from torch.utils.data import DataLoader

class Prebatcher(object):

    def __init__(self, loader, interval, num_workers=2, prefetch_factor=1, pin_memory=True, persistent_workers=True, shuffle=True):
        self.loader = loader
        self.interval = interval
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.prebatched_loader = self.prebatch(loader, num_workers, prefetch_factor, pin_memory, persistent_workers, shuffle)

    @staticmethod
    def prebatch(loader, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True, shuffle=True, shutdown_loader=False):
        data_list = [batch for batch in loader]
        prebatched_loader = DataLoader(
            data_list,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=lambda x: x[0],
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers)
        if shutdown_loader:
            del loader
        return prebatched_loader

    def refresh(self, current_epoch):
        if current_epoch % self.interval == 0:
            self.prebatched_loader = self.prebatch(self.loader, self.num_workers, self.prefetch_factor, self.pin_memory, self.persistent_workers, self.shuffle)
        return self.prebatched_loader

    def shutdown(self):
        del self.loader
        del self.prebatched_loader