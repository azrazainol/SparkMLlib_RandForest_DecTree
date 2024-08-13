# Random Forest and Decision Tree models for predicting Iris Plant Class
## STQD6324_Assignment_03

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

The data is converted into numerical vector format using `VectorAssembler` so that the model can interpret the data. The data is splitted into 70% training data and 30% testing data.

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

### Stop Spark Session

The Spark session is stopped to end Python interaction with Spark.

```python
# Stop the Spark session
spark.stop()
```


### Results and Discussion

#### Model Parameters

The best model defined by the cross-validation method for the Random Forest and Decision Tree are shown and explained below:

**Random Forest**
```
Random Forest Parameters
numTrees: 10
maxDepth: 5
bootstrap: True
featureSubsetStrategy: auto
impurity: gini
```
- The parameters for the Random Forest model shows that the best model contains 10 trees and has a maximum depth of 5. This means that 10 trees was found to be the optimal number of trees in the forest and the maximum number of times the tree splits is 5.
- The bootstrap parameter indicates that each tree in the forest was trained using a random set of samples from the training dataset.
- The feature subset strategy indicates the number of features to use for each split in the tree. The `auto` means that the number of features at each split is determined by using the squareroot of the number of features.
- The impurity parameter indicates the method of measure for the quality of the splits. A lower impurity indicates better quality splits. This model uses `gini` which is a commonly used method for classification models and helps in selecting the best splits to create homogenous subsets.

**Decison Tree**
```
Decision Tree Parameters
maxDepth: 5
impurity: gini
```
- The parameters for the Decision Tree model shows that the best model contains a maximum depth of 5, similar to the Random Forest model. This means that the tree may contain a maximum of 5 splits between the root node and the leaf (final node).
- The impurity parameter is set to `gini` which is the same as the Random forest model.

#### Model Evaluation Metrics

The model evaluation metrics are shown below:

**Random Forest**
```
Random Forest Metrics:
Accuracy: 0.9821428571428571
Precision: 0.9835164835164836
Recall: 0.9821428571428572
F1 Score: 0.9822586872586874

Confusion Matrix (Random Forest):
 [[25  0  0]
 [ 0 12  0]
 [ 0  1 18]]
```
**Decison Tree**
```
Decision Tree Metrics:
Accuracy: 0.9821428571428571
Precision: 0.9835164835164836
Recall: 0.9821428571428572
F1 Score: 0.9822586872586874

Confusion Matrix (Decision Tree):
 [[25  0  0]
 [ 0 12  0]
 [ 0  1 18]]
```

It can be seen that both models outcome for all metrics are the same. The reason both models have similar outcomes may be due to the size of the dataset being small. One of the rules of a good model is to be as simple as possible while retaining high performance. In this case, both models are equally based on the evaluation metrics. Due to that, the Decision Tree model would be the better model to use for future predictions as the results are the same despite using less computational cost.

Based on the results of the evaluation metrics, the accuracy of the model is 0.982 which means that the model classified 98.2% of the testing dataset correctly.
Then the precision metric, 0.983, shows that the proportion of true positive predictions against the total positive predictions for this model is 98.3%.
The recall metric, 0.982, shows that the the proportion of true positive predictions against the actual number of positive instances for this model is 98.2%.
The F1 score metric shows the combined value of the precision and recall metrics to create a more balanced evaluation metric, making sure that it considers any imbalance among classes. The model has an F1 score of 0.982 which means that the balance between precision and recall for this model is 98.2%.
Based on the Confusion Matrix, the model only misclassified once which is a Virginica type wrongly classified as Versicolor type.

To conclude, the simpler model, the Decision Tree model, is chosen as the better model because it produces the same outcome while also using minimal computational costs. With an accuracy and F1 score both at 98.2%, the model proves to be performing well.
