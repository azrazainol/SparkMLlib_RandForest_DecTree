# STQD6324_Assignment_03

This assignment uses Spark MLlib to train a classification model on the Iris dataset.

The first part of the code imports the Iris dataset

```python:Assignment3_P137262.ipynb
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

# Convert to Spark DataFrame
df = spark.createDataFrame(iris_df)

```




