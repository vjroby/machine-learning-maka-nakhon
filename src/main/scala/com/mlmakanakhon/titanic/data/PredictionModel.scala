package com.mlmakanakhon.titanic.data

import org.apache.spark.sql.DataFrame

trait PredictionModel {
  def prediction(dataframe:DataFrame) : DataFrame
}
