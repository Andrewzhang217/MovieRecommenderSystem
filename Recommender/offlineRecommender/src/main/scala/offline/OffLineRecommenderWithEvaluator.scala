package offline

import org.apache.hadoop.shaded.org.eclipse.jetty.websocket.common.frames.DataFrame
import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, LongType}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.jblas.DoubleMatrix
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

case class MovieRating(uid: Int, mid: Int, score: Float, timestamp: Long)
case class Movie(mid: Int, name: String, genres: String)
case class Recommendation(mid: Int, score:Double)
case class UserRecs(uid: Int, recs: Seq[Recommendation])
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object OfflineRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val USER_MAX_RECOMMENDATION = 10
  val MOVIE_MAX_RECOMMENDATION = 10


  def main(args: Array[String]): Unit = {



    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster("local[*]")
      .set("spark.executor.memory", "6G").set("spark.driver.memory", "3G")

    // SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()


    import spark.implicits._

    val moviesPath = "Recommender/offlineRecommender/src/main/resources/movies1.csv"
    val ratingsPath = "Recommender/offlineRecommender/src/main/resources/ratings1.csv"
    val movieRecPath = "Recommender/offlineRecommender/src/main/resources/movieRec"
    val ratingRDD = spark
      .read
      .option("header", "true").csv(ratingsPath)
      .withColumn("uid", col("userId").cast(IntegerType))
      .withColumn("mid", col("movieId").cast(IntegerType))
      .withColumn("score", col("rating").cast(FloatType))
      .withColumn("timestamp", col("timestamp").cast(LongType))
      .as[MovieRating]
      .map(rating => (rating.uid, rating.mid, rating.score, rating.timestamp))
      .cache()


    ratingRDD.toDF().show(7)
    val userRDD = ratingRDD.map(_._1).distinct()

    var movieRDD = spark
      .read
      .option("header", "true").csv(moviesPath)
      .withColumn("mid", col("movieId").cast(IntegerType))
      .withColumn("name", col("title"))
      .as[Movie]
      .rdd
      .map(_.mid)
      .cache()

    movieRDD.toDF().show(6)
    val Array(training, test) = ratingRDD.randomSplit(Array(0.8, 0.2))

    val trainData = training.map(x => Rating(x._1, x._2, x._3)).toDF()
    val testData = test.map(x => Rating(x._1, x._2, x._3)).toDF()
    val als = new ALS()
      .setMaxIter(6)
      .setRank(2)
      .setRegParam(0.01)
      .setUserCol("user")
      .setItemCol("item")
      .setRatingCol("rating")

    val model = als.fit(trainData)

    model.setColdStartStrategy("drop")
    val predictions = model.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    System.out.println("yy" + rmse.toString)
    println(s"Root-mean-square error = $rmse")

    val userRecs = model.recommendForAllUsers(10)

    userRecs.printSchema()
    userRecs.repartition(1).write.format("json").mode("overwrite").save("Recommender/offlineRecommender/src/main/resources/userrec2.json")

    val movieRecs = model.recommendForAllItems(10)
   
    movieRecs.printSchema()
    movieRecs.repartition().write.format("json").mode("overwrite").save("Recommender/offlineRecommender/src/main/resources/movierec2.json")

    spark.stop()
    // train ALS model

  }
}

