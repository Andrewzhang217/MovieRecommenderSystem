package offline

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.functions.{col, struct, to_json}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.jblas.DoubleMatrix

import java.io.{BufferedWriter, File, FileWriter}

case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: String)

case class Movie(mid: Int, name: String, genres: String)

case class Recommendation(mid: Int, score: Double)

case class UserRecs(uid: Int, recs: Seq[Recommendation])

object OfflineRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val USER_MAX_RECOMMENDATION = 20

  case class MovieRecs(mid: Int, recs: Seq[Recommendation])

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val moviesPath = "Recommender/offlineRecommender/src/main/resources/movies.csv"
    val ratingsPath = "Recommender/offlineRecommender/src/main/resources/ratings.csv"
    val movieRecPath = "Recommender/offlineRecommender/src/main/resources/movieRec"
    val ratingRDD = spark
      .read
      .option("header", "true").csv(ratingsPath)
      .withColumn("uid", col("userId").cast(IntegerType))
      .withColumn("mid", col("movieId").cast(IntegerType))
      .withColumn("score", col("rating").cast(DoubleType))
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score))
      .cache()


    //ratingRDD.toDF().show(5)
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

    //movieRDD.toDF().show(5)
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))
    val (rank, iterations, lambda) = (50, 1, 0.1)

    val model = ALS.train(trainData, rank, iterations, lambda)

    val userMovies = userRDD.cartesian(movieRDD)

    // 预测值
    val preRatings = model.predict(userMovies)

    val userRecs = preRatings
      .filter(x => x.rating > 0.6)
      .map(rating => (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map {
        case (uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2) // 降序排序
          .take(USER_MAX_RECOMMENDATION).map(x => Recommendation(x._1, x._2)))
      }.toDF()

    //userRecs.printSchema()
    //    userRecs.repartition(1).write.mode("overwrite").saveAsTable("userRec")
    val movieFeatures = model.productFeatures.map {
      case (mid, features) => (mid, new DoubleMatrix(features))
    }
    // 过滤
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter {
        case (x, y) => x._1 != y._1
      }
      .map {
        case (x, y) =>
          val simScore = consinSim(x._2, y._2)
          (x._1, (y._1, simScore))
      }
      .filter(_._2._2 > 0.6)
      .groupByKey()
      .map {
        case (mid, item) => MovieRecs(mid, item.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()

    val flattenedRecs = movieRecs.withColumn("exploded", functions.explode(col("recs")))
      .withColumn("midScore", to_json(col("exploded")))
      .select("mid", "midScore")

    flattenedRecs.printSchema()

    flattenedRecs
      .write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .csv("Recommender/offlineRecommender/src/main/resources/temp")

    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)

    val srcPath=new Path("Recommender/offlineRecommender/src/main/resources/temp")
    val destPath= new Path("Recommender/offlineRecommender/src/main/resources/movieRec.csv")
    val srcFile=FileUtil.listFiles(new File("Recommender/offlineRecommender/src/main/resources/temp"))
      .filter(f=>f.getPath.endsWith(".csv"))(0)
    FileUtil.copy(srcFile,hdfs,destPath,true,hadoopConfig)
    //Removes CRC File that create from above statement
    hdfs.delete(new Path(".movieRec.csv.crc"),true)
    //Remove Directory created by df.write()
    hdfs.delete(srcPath,true)

    spark.stop()
  }

  def consinSim(x: DoubleMatrix, y: DoubleMatrix): Double = {
    x.dot(y) / (x.norm2() * y.norm2())
  }
}