package com.mlmakanakhon.titanic.evaluator

import org.apache.spark.sql.{DataFrame, SparkSession}

class EvaluateResults(sparkSession: SparkSession) {

  def run(results: DataFrame): Evaluation = {
    val spark = results.sparkSession
    import spark.implicits._
    results.cache()
    val wrong =  results.where($"prediction" !== $"label").count()
    val count = results.count()
    val correct = results.where($"prediction" === $"label").count()
    Evaluation(
      count,
      correct,
      wrong,
      results.where($"prediction" === 0.0 ).where($"prediction" === $"label").count(),
      results.where($"prediction" === 0.0).where($"prediction" !== $"label").count(),
      results.where($"prediction" === 1.0).where($"prediction" !== $"label").count(),
      wrong.toDouble/count.toDouble,
      correct.toDouble/count.toDouble
    )
  }
}

case class Evaluation(totalResults: Long, correct: Long, wrong: Long, truePositive: Long, falseNegative: Long,
                      falsePositive: Long, ratioWrong: Double, ratioCorrect: Double)

object EvaluateResults {
  def apply(sparkSession: SparkSession): EvaluateResults = new EvaluateResults(sparkSession)
}
