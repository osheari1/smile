package smile.classification

import io.circe.Json
import io.circe.parser._
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
  def testBSplineBasis(): Unit = {
    for {
      inputs <- Utils.readTestData()
      (basis, x, nSplines, splineOrder, edgeKnots) <- inputs
    } {
      val basisTest = bSplineBasis(x, edgeKnots, nSplines, splineOrder)
      basis.zip(basisTest).foreach {
        case (row, rowTest) => assertArrayEquals(row, rowTest, 0.000000001)
      }
    }
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
    (Array[Array[Double]], Array[Double], Int, Int, (Double, Double))

  def randomInputData(): (Array[Double], (Double, Double)) = {
    val rng = new Random(1)
    val x =
      (for { _ <- 10 to rng.nextInt(100) } yield rng.nextDouble()).toArray
    val edgeKnots = (x.min, x.max)
    (x, edgeKnots)
  }

  def readTestData(): Option[Vector[bSplineData]] =
    readJsonFromFile().asArray flatMap { jsons =>
      Some((jsons map extractFromJson) flatMap (_.toList))
    }

  def readJsonFromFile(): Json = {
    val json: Json = parse(
      Source
        .fromResource("classification/LogisticGAM/bSplineTestData.json")
        .getLines()
        .foldLeft("")((acc, x) => acc + x)
    ).getOrElse(Json.Null)

    assert(json != Json.Null, "Could not parse test data.")
    json
  }

  def extractFromJson(json: Json): Option[bSplineData] =
    (
      json.hcursor.get[Array[Array[Double]]]("basis"),
      json.hcursor.get[Array[Double]]("x"),
      json.hcursor.get[Int]("nSplines"),
      json.hcursor.get[Int]("splineOrder"),
      json.hcursor.get[(Double, Double)]("edgeKnots")
    ) match {
      case (Right(basis),
            Right(x),
            Right(nSplines),
            Right(splineOrder),
            Right(edgeKnots)) =>
        Some((basis, x, nSplines, splineOrder, edgeKnots))
      case _ => None
    }

}
