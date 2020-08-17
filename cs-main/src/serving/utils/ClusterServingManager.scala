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

import java.io.PrintWriter
import org.apache.flink.core.execution.JobClient

object ClusterServingManager {

  def writeObjectToFile(cli: JobClient): Unit = {
    try {
      new PrintWriter("/tmp/cluster-serving-job-id") {
        write(cli.getJobID.toHexString)
        close
      }
      println("Cluster Serving Flink job id written to file.")
    }
    catch {
      case e: Exception =>
        e.printStackTrace()
        println("Failed to write job id written to file. You may not manager job by id now.")
    }
  }
}
