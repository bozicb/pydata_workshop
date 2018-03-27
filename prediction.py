import urllib.request
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, Normalizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression

urllib.request.urlretrieve("https://wsleaderboard.herokuapp.com/data/test.snappy.parquet", "/tmp/test.snappy.parquet")
urllib.request.urlretrieve("https://wsleaderboard.herokuapp.com/data/training.snappy.parquet", "/tmp/training.snappy.parquet")
urllib.request.urlretrieve("https://wsleaderboard.herokuapp.com/data/test.snappy.parquet", "/tmp/validation.snappy.parquet")

data = sqlContext.read.parquet("/tmp/training.snappy.parquet")
test = sqlContext.read.parquet("/tmp/test.snappy.parquet")
validation = sqlContext.read.parquet("/tmp/validation.snappy.parquet")

# Balancing the dataset (highly skewed data)
balancedDataset = data.filter("y = 'no'").limit(8000).union(data.filter("y = 'yes'"))

# Model features
jobIndexer = StringIndexer(inputCol="job", outputCol="jobInd")
jobEncoder = OneHotEncoder(inputCol="jobInd", outputCol="jobVec")
assembler = VectorAssembler(inputCols=["jobVec"],outputCol="assembledFeatures")
# Label creation
labelIndexer = StringIndexer(inputCol="y", outputCol="indexedLabel")

# Random forest model
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="assembledFeatures", numTrees=25)
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol="indexedLabel", featuresCol="assembledFeatures")

# Pipeline
pipeline = Pipeline(stages=[jobIndexer, jobEncoder,  assembler, labelIndexer, rf])

model = pipeline.fit(balancedDataset)   # Using the training set
predictions = model.transform(test)     # Using the test set

predictionsAndLabels = predictions.rdd.map(lambda x: (x['indexedLabel'], x['prediction']))
# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionsAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)

results = model.transform(validation).sort("id")
submission = "".join(list(map(lambda x: '1' if x['prediction'] == 1.0 else '0', results.select('prediction').collect())))
