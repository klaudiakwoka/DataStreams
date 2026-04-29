from river import linear_model, tree, preprocessing, compose, forest, ensemble, stats

seed = 15

class ModelFactory:
    def create(self, model_name, transformer=None):
        
        models = {
            "HTR": tree.HoeffdingTreeRegressor(),
            "ARF": forest.ARFRegressor(n_models=100, seed=seed),
            "SRP": ensemble.SRPRegressor(
                model=tree.HoeffdingTreeRegressor(),
                n_models=100,
                seed=seed
            ),
            "MEAN": stats.Mean()
        }

        model = models[model_name]

        scaler = preprocessing.MinMaxScaler()

        if transformer is None:
            return compose.Pipeline(scaler, model)

        return compose.Pipeline(transformer, scaler, model)

