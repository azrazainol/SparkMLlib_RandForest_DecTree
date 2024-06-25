# STQD6324_Assignment_03

This assignment uses Spark MLlib to train a classification model on the Iris dataset.

### Importing Libraries

The first part of the code imports the libraries used for:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

```

### Start Spark

```python
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()
```

### Loading Iris Dataset

Next the Iris dataset is imported into the notebook and converted into a Sprak dataframe:

```python
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# Convert to Spark DataFrame
df = spark.createDataFrame(iris_df)

```

### Format and Split Dataset

```python
# Feature Engineering
assembler = VectorAssembler(inputCols=iris.feature_names, outputCol="features")
df = assembler.transform(df)

# Index labels
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
df = indexer.fit(df).transform(df)

# Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

```

### Random Forest

```python

# Define Random Forest classifier
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", seed=42)

# Set up pipeline for Random Forest
pipeline_rf = Pipeline(stages=[rf])

# Create parameter grid for Random Forest
paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.numTrees, [10, 15, 20, 25, 30])  # Number of trees in the forest
                .addGrid(rf.maxDepth, [5, 7, 10, 12, 15])   # Maximum depth of the tree
                .build())

# Define cross-validator for Random Forest
crossval_rf = CrossValidator(estimator=pipeline_rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="accuracy"),
                             numFolds=5, seed=42)  # Set seed for reproducibility

# Train the Random Forest model using cross-validation
cvModel_rf = crossval_rf.fit(train_data)

# Print best model parameters for Random Forest
best_rf_model = cvModel_rf.bestModel.stages[-1]
print("Random Forest - Best Model Parameters:")
print(best_rf_model.extractParamMap())
print()

# Make predictions on the test data using Random Forest
predictions_rf = cvModel_rf.transform(test_data)

# Evaluate Random Forest model
evaluator_rf = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="accuracy")
accuracy_rf = evaluator_rf.evaluate(predictions_rf)

evaluator_rf.setMetricName("weightedPrecision")
precision_rf = evaluator_rf.evaluate(predictions_rf)

evaluator_rf.setMetricName("weightedRecall")
recall_rf = evaluator_rf.evaluate(predictions_rf)

evaluator_rf.setMetricName("f1")
f1_score_rf = evaluator_rf.evaluate(predictions_rf)

# Print evaluation metrics for Random Forest
print("Random Forest Metrics:")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_score_rf}")
print()

# Compute confusion matrix for Random Forest
y_true_rf = predictions_rf.select("indexedLabel").toPandas()
y_pred_rf = predictions_rf.select("prediction").toPandas()

cm_rf = confusion_matrix(y_true_rf, y_pred_rf)
print("Confusion Matrix (Random Forest):\n", cm_rf)
print()

```

```python

# Define Decision Tree classifier
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", seed=42)

# Set up pipeline for Decision Tree
pipeline_dt = Pipeline(stages=[dt])

# Create parameter grid for Decision Tree
paramGrid_dt = (ParamGridBuilder()
                .addGrid(dt.maxDepth, [5, 7, 10, 12, 15])  # Maximum depth of the tree
                .build())

# Define cross-validator for Decision Tree
crossval_dt = CrossValidator(estimator=pipeline_dt,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="accuracy"),
                             numFolds=5, seed=42)

# Train the Decision Tree model using cross-validation
cvModel_dt = crossval_dt.fit(train_data)

# Print best model parameters for Decision Tree
best_dt_model = cvModel_dt.bestModel.stages[-1]
print("Decision Tree - Best Model Parameters:")
print(best_dt_model.extractParamMap())
print()

# Make predictions on the test data using Decision Tree
predictions_dt = cvModel_dt.transform(test_data)

# Evaluate Decision Tree model
evaluator_dt = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="accuracy")
accuracy_dt = evaluator_dt.evaluate(predictions_dt)

evaluator_dt.setMetricName("weightedPrecision")
precision_dt = evaluator_dt.evaluate(predictions_dt)

evaluator_dt.setMetricName("weightedRecall")
recall_dt = evaluator_dt.evaluate(predictions_dt)

evaluator_dt.setMetricName("f1")
f1_score_dt = evaluator_dt.evaluate(predictions_dt)

# Print evaluation metrics for Decision Tree
print("Decision Tree Metrics:")
print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {precision_dt}")
print(f"Recall: {recall_dt}")
print(f"F1 Score: {f1_score_dt}")
print()

# Compute confusion matrix for Decision Tree
y_true_dt = predictions_dt.select("indexedLabel").toPandas()
y_pred_dt = predictions_dt.select("prediction").toPandas()

cm_dt = confusion_matrix(y_true_dt, y_pred_dt)
print("Confusion Matrix (Decision Tree):\n", cm_dt)
print()

# Stop the Spark session
spark.stop()

```

