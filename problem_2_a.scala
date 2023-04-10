
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.classification.{DecisionTreeClassifier,RandomForestClassifier, RandomForestClassificationModel}

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

//val data = sc.textFile("input-path-to-NOAA-files")
val data = sc.textFile("/home/users/mdislam/myfile/NOAA-LU-1949-2020")

data.take(10).foreach(println)
// filter lines with missing temperature label!
val dataFilter = data.filter(line => line.substring(87, 92) != "+9999") 
dataFilter.take(10).foreach(println)

val parsedData = dataFilter.map { line =>
  val year = line.substring(15, 19).toInt
  val month = line.substring(19, 21).toInt
  val day = line.substring(21, 23).toInt
  val time = line.substring(23,25).toDouble
  val latitude = line.substring(28, 34).toDouble / 1000
  val longitude = line.substring(34, 41).toDouble / 1000
  val elevationDimension = line.substring(46, 51).toDouble
  val directionAngle = line.substring(60, 63).toDouble
  val speedRate = line.substring(65, 69).toDouble / 10
  val ceilingHeightDimension = line.substring(70, 75).toDouble
  val distanceDimension = line.substring(78, 84).toDouble
  val airTemperature = line.substring(87, 92).toDouble/10
  val dewPointTemperature = line.substring(93, 98).toDouble / 10

    (year,month,day,time,latitude,longitude,elevationDimension,directionAngle,speedRate,ceilingHeightDimension,distanceDimension,airTemperature,dewPointTemperature)
 
  //year+","+month+","+day+","+time+","+latitude+","+longitude+","+elevationDimension+","+directionAngle+","+speedRate+","+ceilingHeightDimension+","+distanceDimension+","+dewPointTemperature+","+airTemperature
}
parsedData.take(10).foreach(println)
parsedData.coalesce(1, true).saveAsTextFile("myfile/NOAA-LU-1949-2020-RDD")

val col_name = Seq("year","month","day","time","latitude","longitude","elevationDimension","directionAngle","speedRate","ceilingHeightDimension","distanceDimension","airTemperature","dewPointTemperature")

val parsedDataWithColumn = parsedData.toDF(col_name:_*).withColumn("isLessthan2019", $"year" < 2019)
//parsedDataWithColumn.take(10).foreach(println)


//val Array(trainData, testData) = parsedData.randomSplit(Array(0.9, 0.1))

val trainData = parsedDataWithColumn.filter($"isLessthan2019");
val testData = parsedDataWithColumn.filter(!$"isLessthan2019");

trainData.cache() // subset of dataset used for training
testData.cache() // subset of dataset used for final evaluation ("testing")

trainData.count()
testData.count()



import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(trainData)
val trainingSummary = lrModel.summary
val accuracyOfTrain = trainingSummary.accuracyOfTrain

val predictions = lrModel.transform(testData)

predictions.filter($"year" === lit(2019) && $"month" === lit(12) && $"day" === lit(24) && $"time" === lit(18.00)).select("year","month","day","time","airTemperature").show(1)

predictions.show(5)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("airTemperature").setFeaturesCol("features").setPredictionCol("prediction").setMetricName("accuracyOfTrain")
val accuracy = evaluator.evaluate(predictions)

