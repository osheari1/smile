package smile.classification

import io.circe.parser._
import io.circe.{HCursor, Json}
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
    // Read test data
    val json: Json = parse(
      Source
        .fromResource("classification/LogisticGAM/bSplineTestData.json")
        .getLines()
        .foldLeft("")((acc, x) => acc + x)
    ).getOrElse(Json.Null)

    assert(json != Json.Null, "Could not parse test data.")

    for {
      arr <- json.asArray
      (data, iter) <- arr.zipWithIndex
    } {
      val cursor: HCursor = data.hcursor
      (
        cursor.get[Array[Array[Double]]]("basis"),
        cursor.get[Array[Double]]("x"),
        cursor.get[Int]("nSplines"),
        cursor.get[Int]("splineOrder"),
        cursor.get[(Double, Double)]("edgeKnots")
      ) match {
        case (Right(basis),
        Right(x),
        Right(nSplines),
        Right(splineOrder),
        Right(edgeKnots)) =>
          val basisTest = bSplineBasis(x, edgeKnots, nSplines, splineOrder)
          basis.zip(basisTest).foreach {
            case (row, rowTest) => assertArrayEquals(row, rowTest, 0.000000001)
          }

        case _ =>
          assert(assertion = false, "Failed to decode test data from Json.")
      }
    }
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidNSplines(): Unit = {
    val (x, edgeKnots) = randomInputData()
    val nSplines = 0
    val splineOrder = 1
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidSplineOrder(): Unit = {
    val (x, edgeKnots) = randomInputData()
    val nSplines = 2
    val splineOrder = -1
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }

  def randomInputData(): (Array[Double], (Double, Double)) = {
    val rng = new Random(1)
    val x = (for {_ <- 10 to rng.nextInt(100)} yield rng.nextDouble()).toArray
    val edgeKnots = (x.min, x.max)
    (x, edgeKnots)
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testBSplineBasisInvalidSplineOrderWithNSplines(): Unit = {
    val (x, edgeKnots) = randomInputData()
    val nSplines = 4
    val splineOrder = 4
    bSplineBasis(x, edgeKnots, nSplines, splineOrder)
    ()
  }

}
