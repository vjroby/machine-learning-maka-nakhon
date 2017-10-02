# machine-learning-maka-nakhon
Machine Learning Spark application that calculated the survival prediction for the passengers on Titanic.


```
sbt compile
```

To start the application
Run `RunApp` class

The project can be run from IntelliJ or the jar can be submited in a spark cluster.
When submited in spark a path to .csv file must be provided as the first argument.
```
/spark-submit /path/to/machine-learning-maka-nakhon_2.11-1.0.jar /path/to/titanic.csv
````

