from hgp_lib.populations.initializers.population_initializer import PopulationInitializer


class RandomPInitializer(PopulationInitializer):
    def generate(self):
        return [self.generate_one() for _ in range(self.pop_size)]

    def generate_one(self):
        raise NotImplementedError("TODO: Implement")
