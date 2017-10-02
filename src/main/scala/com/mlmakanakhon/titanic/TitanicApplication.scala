package com.mlmakanakhon.titanic

import com.mlmakanakhon.titanic.data.{ApplyLogisticRegression, RawDataReader}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class TitanicApplication(sparkSession: SparkSession, applyLogisticRegression: ApplyLogisticRegression){

  def run(): Unit = {

    val sqlContext = sparkSession.sqlContext

    val rawDataReader: RawDataReader = RawDataReader(sparkSession)

    val csvDF = rawDataReader.data()

    val csvRdd: RDD[Row] = transformCabin(csvDF)

    val schema = getSchema
    val cabinDF = sparkSession.createDataFrame(csvRdd, schema)

    cabinDF.show()
    val withCabinsDF: DataFrame = csvDF.join(cabinDF, Seq("PassengerId"))
    val predictions = applyLogisticRegression.prediction(withCabinsDF)

    // create an Evaluator for binary classification, which expects two input columns: rawPrediction and label.**
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)

    println("Accuracy: " + accuracy)

    predictions.show()
  }

  private def transformCabin(csvDF: DataFrame) = {
    csvDF.select("PassengerId", "Cabin").rdd.map(row => {
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
  }

  private def getSchema = {
    StructType(Array(
      StructField("PassengerId", LongType, true),
      StructField("Char", StringType, true),
      StructField("CharRest", StringType, true),
      StructField("Cabins", IntegerType, true)
    ))
  }
}

object TitanicApplication {
  def apply(sparkSession: SparkSession): TitanicApplication = {
    new TitanicApplication(
      sparkSession,
      ApplyLogisticRegression(sparkSession)
    )
  }
}

//case class RawTitanic(PassengerId: Long, Survived: Integer, Pclass: Int, Name: String, Sex: String, Age: Int, SibSp: Integer,
//                      Parch: Integer, Ticket: String, Fare: Double, Cabin: String, Embarked: String)

