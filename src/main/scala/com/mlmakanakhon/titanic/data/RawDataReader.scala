package com.mlmakanakhon.titanic.data

import java.io.InputStream

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

class RawDataReader(sparkSession: SparkSession) {
  def data(): DataFrame = {
    this.getClass.getResource("/titanic.csv") match {
      case null => throw new Exception("File not found: resources/titanic.csv")
      case resource => {
        sparkSession.read
          .format("csv")
          .option("header", "true")
          .schema(customSchema())
          .load(resource.getPath)
      }
    }
  }
  def customSchema(): StructType = {
    StructType(Array(
      StructField("PassengerId", LongType, true),
      StructField("Survived", IntegerType, true),
      StructField("Pclass", IntegerType, true),
      StructField("Name", StringType, true),
      StructField("Sex", StringType, true),
      StructField("Age", IntegerType, true),
      StructField("SibSp", IntegerType, true),
      StructField("Parch", IntegerType, true),
      StructField("Ticket", StringType, true),
      StructField("Fare", DoubleType, true),
      StructField("Cabin", StringType, true),
      StructField("Embarked", StringType, true)
    ))
  }
}


object RawDataReader{
  def apply(sparkSession: SparkSession): RawDataReader = new RawDataReader(sparkSession)
}