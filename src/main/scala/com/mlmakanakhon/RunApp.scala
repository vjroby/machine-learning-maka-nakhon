package com.mlmakanakhon

import com.mlmakanakhon.titanic.StartApp
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object RunApp {

  def main(args: Array[String]): Unit = {
    StartApp.main(args)
  }

  def localSpark() : SparkSession = {
    val conf = new SparkConf()
      .setMaster("local[1]")
      .setAppName("My App")
      .set("spark.cores.max", "4")
      .set("spark.executor.memory", "6g")

    SparkSession.builder()
      .config(conf)
      .getOrCreate()
  }
}
