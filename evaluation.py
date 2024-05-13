# evaluation.py
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def evaluate_model(predictions):
    # Check if 'probability' column exists and use it if 'rawPrediction' is unsuitable
    if 'probability' in predictions.columns:
        # Adjust evaluator to use 'probability' column for ROC if needed
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="probability",  # Change to probability if rawPrediction issues persist
            metricName="areaUnderROC"
        )
    else:
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

    try:
        auc = evaluator.evaluate(predictions)
        print(f"Area under ROC: {auc}")
    except Exception as e:
        print(f"Error evaluating model: {e}")

    # Multi-class classification evaluator for accuracy, precision, recall, and f1-score
    evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "precisionByLabel"})
    recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "recallByLabel"})
    f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
