package com.mlmakanakhon.titanic

import com.mlmakanakhon.RunApp
import com.mlmakanakhon.titanic.data.{ApplyLogisticRegression, RawDataReader}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class TitanicApplication(sparkSession: SparkSession, applyLogisticRegression: ApplyLogisticRegression) extends LazyLogging {

  def run(): Unit = {
    logger.info("Starting the Titanic Application")

    val sqlContext = sparkSession.sqlContext

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
    val predictions = applyLogisticRegression.prediction(withCabinsDF)

    predictions.show()
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

case class RawTitanic(PassengerId: Long, Survived: Integer, Pclass: Int, Name: String, Sex: String, Age: Int, SibSp: Integer,
                      Parch: Integer, Ticket: String, Fare: Double, Cabin: String, Embarked: String)

