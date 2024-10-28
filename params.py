class TrainerParams:
    def __init__(self, n_epochs=30, num_workers=0, batch_size=20):
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.batch_size = batch_size