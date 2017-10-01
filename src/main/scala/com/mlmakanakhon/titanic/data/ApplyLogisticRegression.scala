package com.mlmakanakhon.titanic.data

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{round, sqrt, when}
import org.apache.spark.sql.{DataFrame, SparkSession}

class ApplyLogisticRegression(sparkSession: SparkSession) extends LazyLogging{

  def prediction(rawData: DataFrame): DataFrame = {

    import sparkSession.implicits._

    val withFeatures: DataFrame = rawData.withColumn("sqr_fare", round(sqrt("Age"), 6))
      .withColumn("Gender=male", when($"Sex" === "male", 1).otherwise(0))
      .withColumn("Gender=female", when($"Sex" === "female", 1).otherwise(0))
      .withColumn("Embarked=C", when($"Embarked" === "C", 1).otherwise(0))
      .withColumn("Embarked=S", when($"Embarked" === "S", 1).otherwise(0))
      .withColumn("Embarked=Q", when($"Embarked" === "Q", 1).otherwise(0))
      .na.fill(0, Seq("Age", "sqr_fare"))

    withFeatures.show()

    val features = withFeatures.drop("PassengerId", "Fare", "Name", "Sex", "Ticket",
      "Cabin", "Embarked", "Cabins", "CharRest", "Char")

    features.show()
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setThreshold(0.4)
      .setWeightCol("Pclass")


    val featuresCols = Array("Pclass","Age","SibSp","Parch","sqr_fare",
      "Gender=male","Gender=female","Embarked=C","Embarked=S","Embarked=Q")

    val assembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")

    val dataWithFeatures = assembler.transform(features)

    val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")

    val dataWithLabel = labelIndexer.fit(dataWithFeatures).transform(dataWithFeatures)

    dataWithLabel.show()

    val splitSeed = 5043
    val Array(trainData:DataFrame, testData: DataFrame) = dataWithLabel.randomSplit(Array(0.7,0.3), splitSeed)

    // Fit the model
    val lrModel = lr.fit(trainData)
    // Print the coefficients and intercept for logistic regression

    logger.info(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    lrModel.transform(testData)
  }
}

object ApplyLogisticRegression{
  def apply(sparkSession: SparkSession): ApplyLogisticRegression = new ApplyLogisticRegression(sparkSession)
}