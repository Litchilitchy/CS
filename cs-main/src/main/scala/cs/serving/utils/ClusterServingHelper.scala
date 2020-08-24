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


package cs.serving.utils

import java.io.{File, FileInputStream}
import java.util.LinkedHashMap

import cs.frameworks.tensorflow.TFNet
import org.apache.log4j.Logger
import org.yaml.snakeyaml.Yaml
import cs.serving.utils.Conventions.Model

class ClusterServingHelper(_configPath: String = "config.yaml", _modelDir: String = null) {
  type HM = LinkedHashMap[String, String]
  val logger: Logger = Logger.getLogger(getClass)

  val configPath = _configPath
  var inferenceMode: String = null
  var redisHost: String = null
  var redisPort: String = null
  var nodeNum: Int = 1
  var coreNum: Int = 1
  var chwFlag: Boolean = true
  var filter: String = null
  var resize: Boolean = false

  /**
   * model related
   */
  var modelType: String = null
  var weightPath: String = null
  var defPath: String = null
  var modelDir = _modelDir
  /**
   * Initialize the parameters by loading config file
   * create log file, set backend engine type flag
   * create "running" flag, for listening the stop signal
   */
  def initArgs(): Unit = {
    println("Loading config at ", configPath)
    val yamlParser = new Yaml()
    val input = new FileInputStream(new File(configPath))
    val configList = yamlParser.load(input).asInstanceOf[HM]

    // parse model field
    val modelConfig = configList.get("model").asInstanceOf[HM]
    if (modelDir == null) {
      modelDir = getYaml(modelConfig, "path", null).asInstanceOf[String]
    }
    inferenceMode = getYaml(modelConfig, "mode", "").asInstanceOf[String]
    parseModelType()

    /**
     * reserved here to change engine type
     * engine type should be able to change in run time
     * but BigDL does not support this currently
     * Once BigDL supports it, engine type could be set here
     * And also other cs.frameworks supporting multiple engine type
     */

    if (modelType.startsWith("tensorflow")) {
      chwFlag = false
    }
    // parse data field
    val dataConfig = configList.get("data").asInstanceOf[HM]
    val redis = getYaml(dataConfig, "src", "localhost:6379").asInstanceOf[String]
    require(redis.split(":").length == 2, "Your redis host " +
      "and port are not valid, please check.")
    redisHost = redis.split(":").head.trim
    redisPort = redis.split(":").last.trim

    filter = getYaml(dataConfig, "filter", "").asInstanceOf[String]
    resize = getYaml(dataConfig, "resize", true).asInstanceOf[Boolean]

    val paramsConfig = configList.get("params").asInstanceOf[HM]
    coreNum = getYaml(paramsConfig, "core_number", 4).asInstanceOf[Int]
  }

  /**
   * The util of getting parameter from yaml
   * @param configList the hashmap of this field in yaml
   * @param key the key of target field
   * @param default default value used when the field is empty
   * @return
   */
  def getYaml(configList: HM, key: String, default: Any): Any = {
    val configValue: Any = try {
      configList.get(key)
    } catch {
      case _ => null
    }
    if (configValue == null) {
      if (default == null) throw new Error(configList.toString + key + " must be provided")
      else {
        default
      }
    }
    else {
      println(configList.toString + key + " getted: " + configValue)
      configValue
    }
  }
  def loadModel(): Model = {
    try {
      modelType match {
        case "tensorflowFrozenModel" =>
          TFNet(weightPath)
        case "tensorflowSavedModel" => TFNet.fromSavedModel(weightPath, inputs = null, outputs = null)
      }
    } catch {
      case e: Exception => println(s"loading model raise error ${e}")
      throw new Error(e)
    }


  }

  /**
   * To check if there already exists detected defPath or weightPath
   * @param defPath Boolean, true means need to check if it is not null
   * @param weightPath Boolean, true means need to check if it is not null
   */
  def throwOneModelError(modelType: Boolean,
                         defPath: Boolean, weightPath: Boolean): Unit = {

    if ((modelType && this.modelType != null) ||
        (defPath && this.defPath != null) ||
        (weightPath && this.weightPath != null)) {
      logger.error("Only one model is allowed to exist in " +
        "model folder, please check your model folder to keep just" +
        "one model in the directory")

    }
  }


  /**
   * Infer the model type in model directory
   * Try every file in the directory, infer which are the
   * model definition file and model weight file
   */
  def parseModelType(): Unit = {
    /**
     * Download file to local if the scheme is remote
     * Currently support hdfs, s3
     */
    val scheme = modelDir.split(":").head
    val localModelPath = if (scheme == "file" || modelDir.split(":").length <= 1) {
      modelDir.split("file://").last
    } else {
      modelDir
    }

    /**
     * Initialize all relevant parameters at first
     */
    modelType = null
    weightPath = null
    defPath = null

    var variablesPathExist = false

    import java.io.File
    val f = new File(localModelPath)
    val fileList = f.listFiles

    if (fileList == null) {
      println("Your model path provided is empty, please check your model path.")
    }
    // model type is always null, not support pass model type currently
    if (modelType == null) {

      for (file <- fileList) {
        val fName = file.getName
        val fPath = new File(localModelPath, fName).toString
        if (fName.endsWith("caffemodel")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "caffe"
        }
        else if (fName.endsWith("prototxt")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }
        // ckpt seems not supported
        else if (fName.endsWith("pb")) {
          throwOneModelError(true, false, true)
          weightPath = localModelPath
          if (variablesPathExist) {
            modelType = "tensorflowSavedModel"
          } else {
            modelType = "tensorflowFrozenModel"
          }
        }
        else if (fName.endsWith("pt")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "pytorch"
        }
        else if (fName.endsWith("model")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "bigdl"
        }
        else if (fName.endsWith("keras")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "keras"
        }
        else if (fName.endsWith("bin")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "openvino"
        }
        else if (fName.endsWith("xml")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }
        else if (fName.equals("variables")) {
          if (modelType != null && modelType.equals("tensorflowFrozenModel")) {
            modelType = "tensorflowSavedModel"
          } else {
            variablesPathExist = true
          }
        }

      }
      if (modelType == null) logger.error("You did not specify modelType before running" +
        " and the model type could not be inferred from the path" +
        "Note that you should put only one model in your model directory" +
        "And if you do not specify the modelType, it will be inferred " +
        "according to your model file extension name")
    }
    else {
      modelType = modelType.toLowerCase
    }

  }

}
