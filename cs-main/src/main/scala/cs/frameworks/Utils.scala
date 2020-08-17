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

package cs.frameworks

import java.io._
import java.nio.file.attribute.PosixFilePermissions
import java.nio.file.{Path => JPath}

import cs.module.Activity
import cs.module.tensor.Tensor
import scala.collection.mutable
import org.apache.log4j.Logger

private object Utils {

  private val logger = Logger.getLogger(getClass)

  @inline
  def timeIt[T](name: String)(f: => T): T = {
    val begin = System.nanoTime()
    val result = f
    val end = System.nanoTime()
    val cost = end - begin
    logger.debug(s"$name time [${cost / 1.0e9} s].")
    result
  }

  def activity2VectorBuilder(data: Activity):
  mutable.Builder[Tensor[_], Vector[Tensor[_]]] = {
    val vec = Vector.newBuilder[Tensor[_]]
    if (data.isTensor) {
      vec += data.asInstanceOf[Tensor[_]]
    } else {
      var i = 0
      while (i < data.toTable.length()) {
        vec += data.toTable(i + 1)
        i += 1
      }
    }
    vec
  }




  def appendPrefix(localPath: String): String = {
    if (!localPath.startsWith("file://")) {
      if (!localPath.startsWith("/")) {
        throw new Exception("local path must be a absolute path")
      } else {
        "file://" + localPath
      }
    } else {
      localPath
    }
  }


  def createTmpDir(prefix: String = "Zoo", permissions: String = "rwx------"): JPath = {
    java.nio.file.Files.createTempDirectory(prefix,
      PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString(permissions)))
  }
}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}


