package com.mlmakanakhon.titanic.data

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{round, sqrt, when}

class ApplyLinearSupportVectorMachine(sparkSession: SparkSession) extends Logging with PredictionModel {

  override def prediction(rawData: DataFrame): DataFrame = {
    val spark = sparkSession

    import spark.implicits._

    val linearSVC = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)


    val withFeatures: DataFrame = createFeatures(rawData)

    val featuresCols = Array("Pclass", "Age", "SibSp", "Parch", "sqr_fare",
      "Gender=male", "Gender=female", "Embarked=C", "Embarked=S", "Embarked=Q")

    val assembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")
    log.info("Applying features.")

    val dataWithFeatures = assembler.transform(withFeatures)

    val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("label")

    val dataWithLabel = labelIndexer.fit(dataWithFeatures).transform(dataWithFeatures)

    val splitSeed = 5043
    val Array(trainData: DataFrame, testData: DataFrame) = dataWithLabel.randomSplit(Array(0.7, 0.3), splitSeed)

    // Fit the model

    val lsvcModel = linearSVC.fit(trainData)

    log.info(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

    lsvcModel.transform(trainData)
  }

  private def createFeatures(rawData: DataFrame) = {
    import rawData.sparkSession.implicits._
    rawData.withColumn("sqr_fare", round(sqrt("Age"), 6))
      .withColumn("Gender=male", when($"Sex" === "male", 1).otherwise(0))
      .withColumn("Gender=female", when($"Sex" === "female", 1).otherwise(0))
      .withColumn("Embarked=C", when($"Embarked" === "C", 1).otherwise(0))
      .withColumn("Embarked=S", when($"Embarked" === "S", 1).otherwise(0))
      .withColumn("Embarked=Q", when($"Embarked" === "Q", 1).otherwise(0))
      .na.fill(0, Seq("Age", "sqr_fare"))
      .drop("PassengerId", "Fare", "Name", "Sex", "Ticket",
        "Cabin", "Embarked", "Cabins", "CharRest", "Char")

  }
}

object ApplyLinearSupportVectorMachine {
  def apply(sparkSession: SparkSession): ApplyLinearSupportVectorMachine = new ApplyLinearSupportVectorMachine(sparkSession)
}
