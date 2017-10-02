name := "machine-learning-maka-nakhon"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
//libraryDependencies += "com.typesafe.scala-logging" % "scala-logging_2.11" % "3.7.2" exclude("com.typesafe.scala-logging", "scala-logging-slf4j_2.11")

mainClass in Compile := Some("com.mlmakanakhon.RunApp")


