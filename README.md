# STQD6324_Assignment_03

This assignment uses Spark MLlib to train a classification model on the Iris dataset.

### Importing Libraries

The first part of the code imports the libraries used.

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

This part of the code starts the Spark session in Python to use Spark code format so that Python can interact with SQL's functionality.

```python
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()
```

### Loading Iris Dataset

Next the Iris dataset is imported into the notebook as a pandas dataframe and converted into a Spark dataframe so that it can be read by Spark:

```python
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# Convert to Spark DataFrame
df = spark.createDataFrame(iris_df)

```

### Format and Split Dataset

The data is converted into numerical vector format so that the model can interpret the data. Then the data is splitted into 70% training data and 30% testing data.

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

This section of code trains a Random Forest classifier for the dataset. First the parameters of the Random Forest is defined to be tested by the cross-validator. Then the training data is used to train the random forest model and cross-validation is performed to find the best model by.

The parameter grid defined are `[10, 15, 20, 25, 30]` for the number of trees and `[5, 7, 10, 12, 15]` for the maximum tree depth. The cross-validation section will find the best parameters for the Random Forest based on the parameter grid defined. 

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

```

### Decision Tree

After the random forest model was trained, the Decision Tree classifier model was trained to compare the performance of the two models so that the best model between the two can be used for future predictions.

Similarly to the steps performed for the Random Forest model, the parameter grid for the Decision Tree was initially defined with a maximum depth of the tree as `[5, 7, 10, 12, 15]`. Then the model is fed with the training data and cross-validation was performed to get the best model for the Decision Tree model.

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
```

### Results and Discussion


