# STQD6324_Assignment_03

This assignment uses Spark MLlib to train a classification model on the Iris dataset.

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

Next the Iris dataset is imported into the notebook:

```python
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# Convert to Spark DataFrame
df = spark.createDataFrame(iris_df)

```




