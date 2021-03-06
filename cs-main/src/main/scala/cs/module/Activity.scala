/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cs.module

import com.google.protobuf.ByteString

import cs.module.tensor.TensorNumericMath.TensorNumeric
import cs.module.tensor.{T, Table, Tensor}

import scala.reflect._

/**
 * [[Activity]] is a trait which represents
 * the concept of neural input within neural
 * networks. For now, two type of input are
 * supported and extending this trait, which
 * are [[Tensor]] and [[Table]].
 */
trait Activity {
  def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D]

  def toTable: Table

  def isTensor: Boolean

  def isTable: Boolean
}

object Activity {
  /**
   * Allocate a data instance by given type D and numeric type T
   * @tparam D Data type
   * @tparam T numeric type
   * @return
   */
  def allocate[D <: Activity: ClassTag, T : ClassTag](): D = {
    val buffer = if (classTag[D] == classTag[Table]) {
      T()
    } else if (classTag[D] == classTag[Tensor[_]]) {
      if (classTag[Boolean] == classTag[T]) {
        Tensor[Boolean]()
      } else if (classTag[Char] == classTag[T]) {
        Tensor[Char]()
      } else if (classTag[Short] == classTag[T]) {
        Tensor[Short]()
      } else if (classTag[Int] == classTag[T]) {
        Tensor[Int]()
      } else if (classTag[Long] == classTag[T]) {
        Tensor[Long]()
      } else if (classTag[Float] == classTag[T]) {
        Tensor[Float]()
      } else if (classTag[Double] == classTag[T]) {
        Tensor[Double]()
      } else if (classTag[String] == classTag[T]) {
        Tensor[String]()
      }else {
        throw new IllegalArgumentException("Type T activity is not supported")
      }
    } else {
      null
    }
    buffer.asInstanceOf[D]
  }

}
