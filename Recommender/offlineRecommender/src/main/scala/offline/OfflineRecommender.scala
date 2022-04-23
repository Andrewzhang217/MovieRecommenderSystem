package offline

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

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

    val moviesPath = "offlineRecommender/src/main/resources/movies.csv"
    val ratingsPath = "offlineRecommender/src/main/resources/ratings.csv"
    val tagsPath = "offlineRecommender/src/main/resources/tags.csv"

    // train ALS model

  }
}
