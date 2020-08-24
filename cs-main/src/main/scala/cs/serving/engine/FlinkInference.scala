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

package cs.serving.engine

import cs.module.tensor.Tensor
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.log4j.Logger
import cs.serving.postprocessing.PostProcessing
import cs.serving.preprocessing.PreProcessing
import cs.serving.utils.Conventions.Model
import cs.serving.utils.{ClusterServingHelper, Conventions, SerParams}
//KMP_AFFINITY=verbose,granularity=fine,proclist=[0,1,2,3,4,5]
class FlinkInference(params: SerParams)
  extends RichMapFunction[List[(String, String)], List[(String, String)]] {
  var t: Tensor[Float] = null
  var logger: Logger = null
  var pre: PreProcessing = null
  var post: PostProcessing = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    val localModelDir = getRuntimeContext.getDistributedCache
      .getFile(Conventions.SERVING_MODEL_TMP_DIR).getPath
    logger.info(s"Model loaded at executor at path ${localModelDir}")
    val helper = new ClusterServingHelper(_modelDir = localModelDir)
    helper.parseModelType()
    params.model = helper.loadModel()

    pre = new PreProcessing(params)

  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val preProcessed = in.map(item => {
      val uri = item._1
      val input = pre.decodeArrowBase64(item._2)
      (uri, input)

    }).toIterator
    val postProcessed =
      InferenceSupportive.singleThreadInference(preProcessed, params).toList
    val t2 = System.nanoTime()
    logger.info(s"${postProcessed.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${postProcessed.size / ((t2 - t1) / 1e9)}")
    postProcessed
  }
}
object FlinkInference {
  var model: Model = null
}