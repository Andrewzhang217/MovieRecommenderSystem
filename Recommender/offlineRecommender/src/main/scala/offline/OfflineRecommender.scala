package offline

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

case class MongoConfig(uri: String, db: String)

case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int)

object OfflineRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  def main(args: Array[String]): Unit = {

    // SparkConf
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )

    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores"))
      .set("spark.executor.memory", "6G").set("spark.driver.memory", "3G")

    // SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // MongoDBConfig
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._
    val ratingRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating=> ( rating.uid, rating.mid, rating.score ))
      .cache()
    // train ALS model

  }
}
