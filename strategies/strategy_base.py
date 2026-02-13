class Strategy:
    def __init__(self, name): self.name = name
    def score(self, market_data): raise NotImplementedError
