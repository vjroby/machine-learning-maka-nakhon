package com.mlmakanakhon.titanic

import com.mlmakanakhon.RunApp
import com.mlmakanakhon.titanic.data.RawDataReader
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object StartApp {
  def main(args: Array[String]): Unit = {


    val sparkSession: SparkSession = RunApp.localSpark()

    val sqlContext = sparkSession.sqlContext

    //    import sqlContext.implicits._

    import sparkSession.implicits._

    val rawDataReader: RawDataReader = RawDataReader(sparkSession)

    val csvDF = rawDataReader.data()




    val csvRdd: RDD[Row] = csvDF.select("PassengerId", "Cabin").rdd.map(row => {
      val cabinChar = row.getAs[String]("Cabin") match {
        case null => (row.getAs[Long]("PassengerId"), "X", "-1", 0)
        case cabin: String => {
          val numberCabins = cabin.split(" ") match {
            case arr =>
              (row.getAs[Long]("PassengerId"), arr.head.substring(0, 1), arr.head.substring(1), arr.length)
          }
          numberCabins
        }
      }
      Row(cabinChar._1, cabinChar._2, cabinChar._3, cabinChar._4)
    }
    )

    val schema = StructType(Array(
      StructField("PassengerId", LongType, true),
      StructField("Char", StringType, true),
      StructField("CharRest", StringType, true),
      StructField("Cabins", IntegerType, true)
    ))
    val cabinDF = sparkSession.createDataFrame(csvRdd, schema)


    cabinDF.show()
    val withCabinsDF: DataFrame = csvDF.join(cabinDF, Seq("PassengerId"))

    val withFeatures: DataFrame = withCabinsDF.withColumn("sqr_fare", round(sqrt("Age"), 6))
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

    val predictions = lrModel.transform(testData)

    predictions
//      .where(col("Survived") === 1)
      .show(200)
    

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
  }

}


case class RawTitanic(PassengerId: Long, Survived: Integer, Pclass: Int, Name: String, Sex: String, Age: Int, SibSp: Integer,
                      Parch: Integer, Ticket: String, Fare: Double, Cabin: String, Embarked: String)

