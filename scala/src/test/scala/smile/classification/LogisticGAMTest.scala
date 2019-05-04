package smile.classification

import io.circe.parser._
import io.circe.{Decoder, Json}
import org.junit.Assert._
import org.junit.Test
import smile.classification.LogisticGAM.BSpline.bSplineBasis

import scala.io.Source
import scala.util.Random

/**
  * @author Riley O'Shea
  */
class LogisticGAMTest {

  @Test
  def testBSplineBasis(): Unit = Utils.readBSplineTestData() match {
    case Some((basis, x, edgeKnots, nSplines, splineOrder)) =>
      for {
        i <- basis.indices
      } {
        val basisTest =
          bSplineBasis(x(i), edgeKnots(i), nSplines(i), splineOrder(i))
        basis(i).zip(basisTest).foreach {
          case (row, rowTest) => assertArrayEquals(row, rowTest, 0.000000001)
        }
      }
    case None => assert(assertion = false, "Could not parse json test data.")
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidNSplines(): Unit = {
    val (x, edgeKnots) = Utils.randomInputData()
    val nSplines = 0
    val splineOrder = 1
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidSplineOrder(): Unit = {
    val (x, edgeKnots) = Utils.randomInputData()
    val nSplines = 2
    val splineOrder = -1
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidSplineOrderWithNSplines(): Unit = {
    val (x, edgeKnots) = Utils.randomInputData()
    val nSplines = 4
    val splineOrder = 4
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }
}

private object Utils {

  type bSplineData =
    (Vector[Array[Array[Double]]],
     Vector[Array[Double]],
     Vector[(Double, Double)],
     Vector[Int],
     Vector[Int])

  def randomInputData(): (Array[Double], (Double, Double)) = {
    val rng = new Random(1)
    val x = (for { _ <- 10 to rng.nextInt(100) } yield rng.nextDouble()).toArray
    val edgeKnots = (x.min, x.max)
    (x, edgeKnots)
  }

  def readBSplineTestData(): Option[bSplineData] = {
    implicit val jsons: Option[Vector[Json]] = readJsonFromFile()
    (
      extractFromJson[Array[Array[Double]]]("basis"),
      extractFromJson[Array[Double]]("x"),
      extractFromJson[(Double, Double)]("edgeKnots"),
      extractFromJson[Int]("nSplines"),
      extractFromJson[Int]("splineOrder"),
    ) match {
      case (Some(basis),
            Some(x),
            Some(edgeKnots),
            Some(nSplines),
            Some(splineOrder)) =>
        Some((basis, x, edgeKnots, nSplines, splineOrder))
      case _ => None
    }
  }

  def extractFromJson[A](key: String)(
      implicit decoder: Decoder[A],
      source: Option[Vector[Json]]): Option[Vector[A]] = {
    source flatMap { jsons =>
      Some {
        jsons map { json =>
          json.hcursor.get[A](key) match {
            case Right(x) => Some(x)
            case _        => None
          }
        } flatMap (_.toList)
      }
    }
  }

  def readJsonFromFile(): Option[Vector[Json]] = {
    val json = parse(
      Source
        .fromResource("classification/LogisticGAM/bSplineTestData.json")
        .getLines()
        .foldLeft("")((acc, x) => acc + x)
    ).getOrElse(Json.Null)

    assert(json != Json.Null, "Could load test data from file.")
    json.asArray
  }
}
