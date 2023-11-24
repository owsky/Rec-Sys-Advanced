from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from scipy.sparse import coo_array
from pyspark.ml.evaluation import RegressionEvaluator


def pyspark_als(coo_array: coo_array, test_data: coo_array):
    """
    Trains the ALS recommender provided by PySpark and prints accuracy metrics
    """
    conf = SparkConf().setAppName("PySpark ALS").setMaster("local")
    spark = SparkSession.Builder().config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Define the schema for the train DataFrame
    schema = StructType(
        [
            StructField("user", IntegerType(), True),
            StructField("item", IntegerType(), True),
            StructField("rating", IntegerType(), True),
        ]
    )

    # Create DataFrames for both train and test data
    train_df = spark.createDataFrame(
        zip(
            coo_array.row.tolist(),
            coo_array.col.tolist(),
            coo_array.data.tolist(),
        ),
        schema=schema,
    )
    test_df = spark.createDataFrame(
        zip(
            test_data.row.tolist(),
            test_data.col.tolist(),
            test_data.data.tolist(),
        ),
        schema=schema,
    )

    als = ALS(
        rank=10,
        maxIter=10,
        regParam=0.01,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
    )

    model = als.fit(train_df)
    predictions = model.transform(test_df)

    # Ignore the nan results obtained from cold users/items
    predictions = predictions.dropna()

    # Create an instance of RegressionEvaluator for MAE
    mae_evaluator = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction"
    )

    # Compute the MAE
    mae = mae_evaluator.evaluate(predictions)

    # Create an instance of RegressionEvaluator for RMSE
    rmse_evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    # Compute the RMSE
    rmse = rmse_evaluator.evaluate(predictions)

    print(f"PySpark MAE: {mae}, PySpark RMSE: {rmse}")

    # Kill the Spark application
    spark.sparkContext.stop()
