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

import java.nio.ByteOrder

import cs.module.tensor.{T, Table, Tensor}
import cs.module.tensor.TensorNumericMath.TensorNumeric
import cs.module.utils.{InferShape, OptimMethod}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * [[TensorModule]] is an abstract sub-class of [[AbstractModule]], whose
 * input and output type both are [[Tensor]].
 *
 * @tparam T The numeric type in this cs.module parameters
 */
abstract class TensorModule[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T]

/**
 * Module is the basic component of a neural network. It forward activities and backward gradients.
 * Modules can connect to others to construct a complex neural network.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T The numeric type in this cs.module parameters.
 */
abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](
                                                                                              implicit ev: TensorNumeric[T]) extends Serializable with InferShape{

  // ================================= Public APIs =============================================


  /**
   * The cached output. So we don't compute it again when need it
   */
  var output: B = Activity.allocate[B, T]()

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  var gradInput: A = Activity.allocate[A, T]()

  /**
   * Get the scale of gradientWeight
   */
  final def getScaleW(): Double = {
    scaleW
  }

  /**
   * Get the scale of gradientBias
   */
  final def getScaleB(): Double = {
    scaleB
  }

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  def setScaleW(w: Double): this.type = {
    scaleW = w
    this
  }

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  def setScaleB(b: Double): this.type = {
    scaleB = b
    this
  }

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  def clearState() : this.type = {
    if (output.isInstanceOf[Tensor[_]]) {
      output.asInstanceOf[Tensor[_]].set()
    }

    if (gradInput.isInstanceOf[Tensor[_]]) {
      gradInput.asInstanceOf[Tensor[_]].set()
    }

    this
  }

  /**
   * Whether user set a name to the cs.module before
   * @return
   */
  final def hasName: Boolean = name != null

  /**
   * Set the cs.module name
   *
   * @param name
   * @return
   */
  final def setName(name : String) : this.type = {
    this.name = name
    this
  }

  /**
   * Get the cs.module name, default name is className@namePostfix
   *
   * @return
   */
  final def getName() : String = {
    if (this.name == null) {
      s"${this.getClass.getSimpleName}${namePostfix}"
    } else {
      this.name
    }
  }

  override def toString(): String = getPrintName

  /**
   * Get the forward/backward cost time for the cs.module or its submodules
   * @return
   */
  def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    Array((this, forwardTime, backwardTime))
  }

  /**
   * Get the forward/backward cost time for the cs.module or its submodules
   * and group by cs.module type.
   * @return (cs.module type name, forward time, backward time)
   */
  final def getTimesGroupByModuleType():
  Array[(String, Long, Long)] = {
    this.getTimes().map(v => (v._1.getClass().getName(), v._2, v._3)).groupBy(_._1)
      .map(v => (v._1, v._2.reduce((a, b) => (v._1, a._2 + b._2, a._3 + b._3))))
      .map(v => (v._1, v._2._2, v._2._3))
      .toArray
      .sortWith((a, b) => (a._2 + a._3) > (b._2 + b._3))
  }

  /**
   * Reset the forward/backward record time for the cs.module or its submodules
   * @return
   */
  def resetTimes(): Unit = {
    forwardTime = 0
    backwardTime = 0
  }

  /**
   * freeze the cs.module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  def freeze(names: String*): this.type = {
    if (names.isEmpty) {
      // in case when freeze is called many times
      if (scaleW != 0) {
        scaleWCache = scaleW
        scaleW = 0
      }
      if (scaleB != 0) {
        scaleBCache = scaleB
        scaleB = 0
      }
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.freeze()
          case _ => throw new Exception(s"cannot match cs.module named $name")
        }
      })
    }
    this
  }

  /**
   * "unfreeze" cs.module, i.e. make the cs.module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of cs.module names to unFreeze
   */
  def unFreeze(names: String*): this.type = {
    if (names.isEmpty) {
      scaleW = scaleWCache
      scaleB = scaleBCache
    } else {
      names.foreach(name => {
        this (name) match {
          case Some(x) => x.unFreeze()
          case _ => throw new Exception(s"cannot match cs.module named $name")
        }
      })
    }
    this
  }

  /**
   * Takes an input object, and computes the corresponding output of the cs.module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  final def forward(input: A): B = {
    val before = System.nanoTime()
    try {
      updateOutput(input)
    } catch {
      case e: Throwable =>
        throw new Error(this.toString(), e)
    }
    forwardTime += System.nanoTime() - before

    output
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  def updateOutput(input: A): B

  /**
   * Computing the gradient of the cs.module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  def updateGradInput(input: A, gradOutput: B): A

  /**
   * Computing the gradient of the cs.module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is cs.module dependent. The cs.module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   */
  def accGradParameters(input: A, gradOutput: B): Unit = {}

  /**
   * If the cs.module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  def zeroGradParameters(): Unit = {
    if (parameters() != null) {
      parameters()._1.zip(parameters()._2)foreach{ case (weight, grad) =>
        grad.resizeAs(weight).zero()
      }
    }
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  /**
   * Get extra parameter in this cs.module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  def getExtraParameter(): Array[Tensor[T]] = null

  /**
   * Set extra parameter to this cs.module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  final def setExtraParameter(extraParam: Array[Tensor[T]]): this.type = {
    val currentExtraParam = this.getExtraParameter()
    if (extraParam != null && currentExtraParam != null) {
      require(extraParam.length == currentExtraParam.length,
        "state's length doesn't match, excepted:" +
          s"${currentExtraParam.length}, but got  ${extraParam.length}")
      var i = 0
      while (i < extraParam.length) {
        currentExtraParam(i).copy(extraParam(i))
        i += 1
      }
      this
    } else if (extraParam == null && currentExtraParam == null) {
      this
    } else {
      throw new IllegalArgumentException(s"cs.module's extraParameter is $currentExtraParam" +
        s", while setting param is ${extraParam}")
    }
  }

  /**
   * This function returns a table contains ModuleName, the parameter names and parameter value
   * in this cs.module.
   *
   * The result table is a structure of Table(ModuleName -> Table(ParameterName -> ParameterValue)),
   * and the type is Table[String, Table[String, Tensor[T]]].
   *
   * For example, get the weight of a cs.module named conv1:
   *   table[Table]("conv1")[Tensor[T]]("weight").
   *
   * The names of the parameters follow such convention:
   *
   * 1. If there's one parameter, the parameter is named as "weight", the gradient is named as
   * "gradWeight"
   *
   * 2. If there're two parameters, the first parameter is named as "weight", the first gradient is
   * named as "gradWeight"; the second parameter is named as "bias", the seconcd gradient is
   * named as "gradBias"
   *
   * 3. If there're more parameters, the weight is named as "weight" with a seq number as suffix,
   * the gradient is named as "gradient" with a seq number as suffix
   *
   * Custom modules should override this function the default impl if the convention doesn't meet
   * the requirement.
   *
   * @return Table
   */
  def getParametersTable(): Table = {
    val params = parameters()
    if (params == null) return null
    val (weights, gradients) = params
    require(gradients.length == weights.length, "weight number is not equal to grad number")

    if (weights.length == 1) {
      T(getName() -> T("weight" -> weights(0), "gradWeight" -> gradients(0)))
    } else if (weights.length == 2) {
      T(getName() -> T("weight" -> weights(0), "bias" -> weights(1),
        "gradWeight" -> gradients(0), "gradBias" -> gradients(1)))
    } else {
      val result = T()
      weights.zip(gradients).zipWithIndex.map { case ((w, g), i) =>
        result(s"weight$i") = w
        result(s"gradient$i") = g
      }
      T(getName() -> result)
    }
  }

  /**
   * Set the cs.module to training mode
   * @return
   */
  def training(): this.type = {
    train = true
    this
  }

  /**
   * Set the cs.module to evaluate mode
   * @return
   */
  def evaluate(): this.type = {
    train = false
    this
  }

  /**
   * Check if the model is in training mode
   * @return
   */
  final def isTraining(): Boolean = {
    this.train
  }

  /**
   * Reset cs.module parameters, which is re-initialize the parameter with given initMethod
   */
  def reset(): Unit = {}

  /**
   * Set the line separator when print the cs.module
   * @param line
   * @return
   */
  final def setLine(line: String): this.type = {
    this.line = line
    this
  }



  override def equals(other: Any): Boolean = other match {
    case that: AbstractModule[A, B, T] =>
      (that canEqual this) &&
        (that.getClass equals this.getClass) &&
        output == that.output &&
        gradInput == that.gradInput &&
        name == that.name
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Object): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(output, gradInput, this.getClass, this.name)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }



  /**
   * Set weight and bias for the cs.module
   * @param newWeights array of weights and bias
   * @return
   */
  final def setWeightsBias(newWeights: Array[Tensor[T]]): this.type = {
    require(parameters() != null, "this layer does not have weight/bias")
    require(parameters()._1.length == newWeights.length,
      "the number of input weight/bias is not consistant with " +
        "number of weight/bias of this layer, " +
        s"number of input ${parameters()._1.length}," +
        s" number of output ${newWeights.length}")
    val weights = parameters()._1
    for(i <- newWeights.indices) {
      // TODO: enable this checking as we don't respect shape right now.
      //      require(weights(i).size().deep == newWeights(i).size().deep,
      //        s"Mismatch shape, ${weights(i).size().mkString(",")}" +
      //          s" vs ${newWeights(i).size().mkString(",")} ")
      weights(i).copy(newWeights(i))
    }
    this
  }

  /**
   * Get weight and bias for the cs.module
   * @return array of weights and bias
   *
   */
  final def getWeightsBias(): Array[Tensor[T]] = {
    if (parameters() != null) {
      parameters()._1
    } else {
      null
    }
  }


  protected def processInputs(nodes: Seq[ModuleNode[T]]): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    nodes.foreach(node => {
      node.add(curNode, Edge())
    })
    curNode
  }

  protected def processInputs(first: (ModuleNode[T], Int),
                              nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    first._1.add(curNode, Edge(first._2))
    nodesWithIndex.foreach(nodeWithIndex => {
      nodeWithIndex._1.add(curNode, Edge(nodeWithIndex._2))
    })
    curNode
  }

  /**
   * Build graph: some other modules point to current cs.module
   * @param nodes upstream cs.module nodes
   * @return node containing current cs.module
   */
  def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    processInputs(nodes)
  }

  /**
   * Build graph: some other modules point to current cs.module
   * @param nodes upstream cs.module nodes in an array
   * @return node containing current cs.module
   */
  def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    processInputs(nodes)
  }

  /**
   * Build graph: some other modules point to current cs.module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream cs.module nodes and the output tensor index. The start index is 1.
   * @return node containing current cs.module
   */
  def inputs(first: (ModuleNode[T], Int), nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    validateInput(List(first._1.element))
    validateInput(nodesWithIndex.map(_._1.element))
    processInputs(first, nodesWithIndex: _*)
  }

  /**
   * Find a cs.module with given name. If there is no cs.module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   * @param name
   * @return
   */
  def apply(name : String): Option[AbstractModule[A, B, T]] = {
    if (this.getName() == name) {
      Some(this)
    } else {
      None
    }
  }



  // ================================= Internal APIs ===========================================

  private var namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  final private def getNamePostfix : String = namePostfix

  final private def setNamePostfix(namePostfix : String) : Unit =
    this.namePostfix = namePostfix

  /**
   * The scale of gradient weight and gradient bias
   * before gradParameters being accumulated.
   */
  protected var scaleW: Double = 1.0
  protected var scaleB: Double = 1.0

  /**
   * The name of the cs.module
   */
  private var name : String = null

  private var id: Int = 0

  private def setId(id: Int): Unit = {
    this.id = id
  }

  private def getId(): Int = this.id

  protected final def getPrintName(): String = {
    val postfix = if (name == null) {
      namePostfix
    } else {
      name
    }
    s"${this.getClass.getSimpleName}[${postfix}]"

  }

  protected var forwardTime = 0L

  protected var backwardTime = 0L

  private var scaleWCache: Double = scaleW
  private var scaleBCache: Double = scaleB

  

  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected var train: Boolean = true


  protected var line = "\n"


  final private def setWeightAndBias(copy : AbstractModule[A, B, T], deepCopy : Boolean): Unit = {
    val parameterTable = this.getParametersTable
    val copiedModuleParamTable = copy.getParametersTable
    if (parameterTable != null) {
      require(copiedModuleParamTable != null, "cloned cs.module should have params")
      parameterTable.foreach {
        case (name: String, params: Table) =>
          require(copiedModuleParamTable.get(name) != None, s"cloned cs.module should have for $name")
          setLayerWeightAndBias(params,
            copiedModuleParamTable.get(name).get.asInstanceOf[Table], deepCopy)
      }
    }
  }

  final private def setLayerWeightAndBias(params : Table,
                                          copyParams : Table, deepCopy : Boolean): Unit = {
    params.foreach(param => {
      copyParam(params, copyParams, deepCopy, param._1.toString)
    })
  }

  final private def copyParam(params : Table, copyParams : Table,
                              deepCopy : Boolean, paraName : String) : Unit = {
    if (params.contains(paraName)) {
      // this is for quantization tensors where the weight might be an array
      if (params.get(paraName).get
        .isInstanceOf[Array[Tensor[T]]]) {
        val copies = copyParams.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        val origins = params.get(paraName).get
          .asInstanceOf[Array[Tensor[T]]]
        var i = 0
        while (i < copies.length) {
          copyTensor(origins(i), copies(i), deepCopy)
          i += 1
        }
      } else {
        // For normal layers, their params are just tensors
        copyTensor(params.get(paraName).get.asInstanceOf[Tensor[T]],
          copyParams.get(paraName).get.asInstanceOf[Tensor[T]], deepCopy)
      }
    }
  }

  final private def copyTensor(t1 : Tensor[T], t2 : Tensor[T], deepCopy : Boolean) = {
    if (deepCopy) {
      t2.copy(t1)
    } else {
      t2.set(t1)
    }
  }

  final private def copyWeights(target: Table, src: Table, matchAll: Boolean): Unit = {
    target.foreach {
      case (name: String, targetParams: Table) =>
        if (src.contains(name)) {
          val srcParams = src[Table](name)
          if (srcParams.contains("weight")) {
            val w = srcParams[Tensor[T]]("weight")
            targetParams[Tensor[T]]("weight").resizeAs(w).copy(w)
          }
          if (srcParams.contains("bias")) {
            val b = srcParams[Tensor[T]]("bias")
            targetParams[Tensor[T]]("bias").resizeAs(b).copy(b)
          }
        } else {
          if (matchAll) new Exception(s"cs.module $name cannot find corresponding weight bias")
        }
    }
  }

  private def canEqual(other: Any): Boolean = other.isInstanceOf[AbstractModule[A, B, T]]


  /**
   * Generate end nodes of current cs.module with start nodes
   * @param startNodes: current start nodes
   * @return current end nodes
   */
  private def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    val endNodes = Array(this.inputs(startNodes: _*))
    endNodes
  }

  /**
   * Return classTag numerics for cs.module serialization. If your cs.module contains multiple classtag
   * in the constructor, you should override this method
   * @return
   */
  private def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T]), Array(ev))
  }

  /**
   * Check if some cs.module is duplicated in the model. For a layer it cannot be duplicated.
   * Container should override this method
   */
  private def checkDuplicate(
                                     record: mutable.HashSet[Int] = mutable.HashSet()
                                   ): Unit = {
    val errMsg = "Some cs.module is duplicate in the current model: "
    val curId = System.identityHashCode(this)
    require(this.skipDuplicateCheck() || !record.contains(curId), errMsg + this.getName())
    record.add(curId)
  }

  /**
   * Sometimes, some layer need skip the duplicate check process, e.g. Keras-like input layer
   * @return
   */
  private def skipDuplicateCheck(): Boolean = false

  /**
   * if the model contains native resources such as aligned memory, we should release it by manual.
   * JVM GC can't release them reliably.
   */
  def release(): Unit = {}




  private var _optimMethod: OptimMethod[T] = null

  /**
   * set optim method
   */

  private def setOptimMethod(optimMethod: OptimMethod[T]): Unit = {
    _optimMethod = optimMethod
  }

  /**
   * get optim method for layer
   */

  private def getOptimMethod(): OptimMethod[T] = _optimMethod
  type ModuleNode[T] = Node[AbstractModule[A, B, T]]
}

