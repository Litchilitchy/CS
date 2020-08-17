/*
 * Copyright 2018 Analytics Zoo Authors.
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

package serving.utils

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType

object Conventions {
  val SERVING_STREAM_NAME = "serving_stream"
  val ARROW_INT = new ArrowType.Int(32, true)
  val ARROW_FLOAT = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
  val ARROW_BINARY = new ArrowType.Binary()
  val ARROW_UTF8 = new ArrowType.Utf8
}
