# tuning.py
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def tune_model(train_df, base_model, param_grid):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    cv = CrossValidator(estimator=base_model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
    model = cv.fit(train_df)
    return model.bestModel
