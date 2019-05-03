/** *****************************************************************************
  * Copyright (c) 2010 Haifeng Li
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  * ******************************************************************************/
package smile.classification

import org.slf4j.{Logger, LoggerFactory}

import scala.annotation.tailrec

object LogisticGAM {
  private val serialVersionUID = 1L
  private val logger: Logger = LoggerFactory.getLogger(LogisticGAM.getClass)

  private[classification] object BSpline {

    def generateEdgeKnots(x: Array[Double], dataType: DataType): (Double, Double) =
      dataType match {
        case Numerical => (x.min, x.max)
        case Categorical => (x.min + 0.5, x.max + 0.5)
      }


    type Extrapolate = (Array[Boolean], Array[Boolean], Array[Boolean])

    /**
      * Generates b-spline basis matrix for a given feature using De Boor algorithm. Basis functions are
      * extrapolated linearly past edge knots.
      *
      * @param x           Input data should correspond to a single feature
      * @param edgeKnots   Locations of 2 edge knots.
      * @param nSplines    Number of splines to generate. Must be >= splineOrder + 1. Default = 20
      * @param splineOrder Order of spline basis. Default = 3
      * @return A matrix containing b-spline basis functions. Will be of the shape (x.length, nSplines)
      */
    private[classification] def bSplineBasis(x: Array[Double],
                                             edgeKnots: (Double, Double),
                                             nSplines: Int = 20,
                                             splineOrder: Int = 3) = {
      if (nSplines < 1) {
        throw new IllegalArgumentException(
          s"Number of splines must be >= 1: $nSplines < 1")
      }

      if (splineOrder < 0) {
        throw new IllegalArgumentException(
          s"Spline order must be >= 1: $splineOrder < 1")
      }

      if (nSplines < splineOrder + 1) {
        throw new IllegalArgumentException(
          s"nSplines must be >= splineOrder + 1:" +
            s"$nSplines < $splineOrder + 1")
      }

      // Rescale edge knots to [0, 1]
      val offset = edgeKnots._1
      val scale = if (edgeKnots._2 - offset == 0) {
        1.0
      } else {
        edgeKnots._2 - offset
      }
      // Creates array of equally spaced values between 0 and 1
      val boundaryKnots = Array.iterate(0.0, 1 + nSplines - splineOrder) { n =>
        val out = n + 1.0 / (nSplines - splineOrder)
        if (out > 1)
          1.0
        else
          out
      }
      val diff = boundaryKnots(1) - boundaryKnots.head

      // Scale X into [0, 1] and add 0.0 and 1.0 to end to get derivatives for extrapolation.
      val xScaled = x.map(v => (v - offset) / scale) ++ Array(0.0, 1.0)

      // Decide which observations will be interpolated and extrapolated
      val extMask = (
        xScaled.map(_ < 0),
        xScaled.map(_ > 1),
        xScaled.map(x => (x >= 0) && (x <= 1))
      )

      // Augmented edge knots
      val aug = (1 to splineOrder).map(_ * diff).toArray
      val augKnots = aug.reverse.map(-_) ++
        boundaryKnots ++
        aug.zipWithIndex.map({
          case (value, i) =>
            if (i == aug.length - 1)
              value + 1 + math.pow(10, -9) // Make last knot inclusive
            else
              value + 1
        })

      // Prep for Haar Basis
      val basis = {
        // Creates basis matrix based on augmented edge knots
        val _basis = Array.tabulate(xScaled.length, augKnots.length - 1) {
          (i, j) =>
            val gt = if (xScaled(i) >= augKnots(j)) 1.0 else 0.0
            val lt = if (xScaled(i) < augKnots(j + 1)) 1.0 else 0.0
            gt * lt
        }
        // Forces basis to be symmetric at 0 and 1
        _basis.update(_basis.length - 1, _basis(_basis.length - 2).reverse)
        _basis
      }

      // Compute basis
      val (computedBasis, prevBasis) = {
        @tailrec
        def loop(n: Int = 2,
                 maxI: Int,
                 basis: Array[Array[Double]],
                 prevBasis: Array[Array[Double]])
        : (Array[Array[Double]], Array[Array[Double]]) =
          n match {
            case _ if n >= splineOrder + 2 => (basis, prevBasis)
            case _ =>
              // Left sub-basis
              val left = {
                val left = Array.tabulate(xScaled.length, maxI) { (i, j) =>
                  (xScaled(i) - augKnots(j)) * basis(i)(j)
                }

                val denominator = augKnots
                  .slice(n - 1, maxI + n - 1)
                  .zip(augKnots.slice(0, maxI))
                  .map({ case (v1, v2) => v1 - v2 })

                // Fill in left sub-basis matrix
                for {
                  row <- left
                  (v, j) <- row.zipWithIndex
                } {
                  row.update(j, v / denominator(j))
                }
                left
              }

              // Right sub-basis
              val right = {
                val _basis = basis.map(_.slice(1, maxI + 1))
                val _augKnots = augKnots.slice(n, maxI + n)

                val right = Array
                  .tabulate(xScaled.length, maxI) { (i, j) =>
                    (_augKnots(j) - xScaled(i)) * _basis(i)(j)
                  }

                val denominator = _augKnots
                  .zip(augKnots.slice(1, maxI + 1))
                  .map({ case (v1, v2) => v1 - v2 })

                for {
                  row <- right
                  (v, j) <- row.zipWithIndex
                } {
                  row.update(j, v / denominator(j))
                }
                right
              }

              // Elementwise add left and right basis
              val nextBasis = left
                .zip(right)
                .map {
                  case (a, b) => a.zip(b).map({ case (v1, v2) => v1 + v2 })
                }

              loop(
                n + 1,
                maxI - 1,
                nextBasis,
                basis.slice(basis.length - 2, basis.length))
          }

        loop(
          2,
          augKnots.length - 2,
          basis,
          Array.fill(basis.length, basis.head.length)(0.0))
      }

      extrapolate(
        xScaled,
        computedBasis,
        prevBasis,
        splineOrder,
        augKnots,
        extMask)
    }

    private def extrapolate(x: Array[Double],
                            basis: Array[Array[Double]],
                            previousBasis: Array[Array[Double]],
                            splineOrder: Int,
                            augKnots: Array[Double],
                            extMask: Extrapolate): Array[Array[Double]] =
      extMask match {
        case (l, r, _)
          if splineOrder <= 0 || (!l.exists(v => v) && !r.exists(v => v)) =>
          basis.slice(0, basis.length - 2)
        case (_, _, it) =>
          // Zero out all basis values that are not interpolated
          val basisZeroed = basis.zipWithIndex
            .map {
              case (arr, i) => if (!it(i)) Array.fill(arr.length)(0.0) else arr
            }

          // Calculate left extrapolation
          val left = {
            val denominator = augKnots
              .slice(splineOrder, augKnots.length - 1)
              .zip(augKnots.slice(0, augKnots.length - splineOrder - 1))
              .map { case (l, r) => l - r }

            previousBasis.map { row =>
              row.slice(0, row.length - 1).zipWithIndex.map {
                case (v, i) => v / denominator(i)
              }
            }
          }

          // Calculate right extrapolation
          val right = {
            val denominator = augKnots
              .slice(splineOrder + 1, augKnots.length)
              .zip(augKnots.slice(1, augKnots.length - splineOrder))
              .map { case (l, r) => l - r }

            previousBasis.map { row =>
              row.slice(1, row.length).zipWithIndex.map {
                case (v, i) => v / denominator(i)
              }
            }
          }

          // Calculate gradients
          val grads = left
            .zip(right)
            .map {
              case (lxs, rxs) =>
                lxs.zip(rxs).map { case (l, r) => (l - r) * splineOrder }
            }

          // Updates a basis matrix from extrapolation mask and computed values
          val updateBasis =
            (basis: Array[Array[Double]],
             e: Array[Boolean],
             values: Array[Array[Double]]) => {
              var ix = 0
              basis.zipWithIndex.collect {
                case (v, i) =>
                  if (e(i)) {
                    ix += 1
                    values(ix - 1)
                  } else {
                    v
                  }
              }
            }

          // Helper function for extrapolating left and right
          val applyExtrapolation =
            (basis: Array[Array[Double]],
             basisIx: Int,
             grad: Array[Double],
             e: Array[Boolean],
             fn: (Double, Double, Double) => Double) => {
              val xExt = x.zipWithIndex.collect({ case (v, i) if e(i) => v })
              Array.tabulate(xExt.length, grad.length) { (i, j) =>
                fn(xExt(i), grad(j), basis(basisIx)(j))
              }
            }

          val basisL = (basis: Array[Array[Double]]) =>
            extMask._1 match {
              case l if l.exists(v => v) =>
                val values = applyExtrapolation(
                  basis,
                  basis.length - 2,
                  grads(0),
                  l,
                  (x, g, b) => x * g + b
                )
                updateBasis(basis, l, values)

              case _ => basis
            }

          val basisR = (basis: Array[Array[Double]]) =>
            extMask._2 match {
              case r if r.exists(v => v) =>
                val values = applyExtrapolation(
                  basis,
                  basis.length - 1,
                  grads(1),
                  r,
                  (x, g, b) => (x - 1) * g + b
                )
                updateBasis(basis, r, values)
              case _ => basis
            }

          basisL.compose(basisR)(basisZeroed)
      }
  }

}
