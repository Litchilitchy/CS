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
package cs.frameworks.tensorflow

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import cs.module.Activity
import cs.module.tensor.Tensor
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.reflect.io.Path
import org.json4s.jackson.JsonMethods._
import org.json4s._

object NetUtils {
  implicit val formats = DefaultFormats
  def processTFFolder(folder: String): (String, Meta) = {
    val folderPath = Path(folder)
    if (!folderPath.exists) {
      throw new IllegalArgumentException(s"$folder does not exist")
    }

    val modelPath = folderPath / Path("frozen_inference_graph.pb")
    if (!modelPath.exists) {
      throw new IllegalArgumentException(
        s"${modelPath.path} does not exist")
    }
    val metaPath = folderPath / Path("graph_meta.json")
    if (!metaPath.exists) {
      throw new IllegalArgumentException(
        s"${metaPath.path} does not exist")
    }

    val jsonStr = Source.fromFile(metaPath.jfile).getLines().mkString

    val meta = parse(jsonStr).camelizeKeys.extract[Meta]
    (modelPath.toString(), meta)
  }
  def generateZeroGrad(input: Activity, grad: Activity): Unit = {
    if (grad.isTable) {
      var i = 0
      while (i < grad.toTable.length()) {
        grad.toTable[Tensor[Float]](i + 1)
          .resizeAs(input.toTable[Tensor[Float]](i + 1))
        i = i + 1
      }
    } else {
      grad.toTensor[Float]
        .resizeAs(input.toTensor[Float])
    }
  }
}

case class Meta(inputNames: Array[String],
                             outputNames: Array[String],
                             tempTensors: Option[Array[String]] = None,
                             variables: Option[Array[String]] = None,
                             gradVariables: Option[Array[String]] = None,
                             gradInputs: Option[Array[String]] = None
                            ) {

  for (name <- inputNames) {
    require(name.split(":").length == 2, s"Input names require to be Tensor names, " +
      s"but <${name}> looks like a operation name, please try <${name}:0> instead.")
  }

  for (name <- outputNames) {
    require(name.split(":").length == 2, s"Output names require to be Tensor names, " +
      s"but <${name}> looks like a operation name, please try <${name}:0> instead.")
  }

}



abstract class SerializationHolder
  extends Serializable with KryoSerializable {

  protected def timing[T](name: String)(f: => T): T = {
    val logger = LoggerFactory.getLogger(getClass)
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = end - begin
    logger.debug(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  protected trait CommonOutputStream {
    def writeInt(value: Int): Unit
    def write(value: Array[Byte]): Unit
    def writeString(value: String): Unit
    def writeObject(obj: AnyRef): Unit
  }

  protected trait CommonInputStream {
    def readInt(): Int
    def read(buff: Array[Byte], off: Int, len: Int): Int
    def skip(len: Int): Unit
    def readString(): String
    def readObject(): AnyRef
  }

  def writeInternal(out: CommonOutputStream): Unit

  def readInternal(in: CommonInputStream): Unit

  private def writeObject(out: java.io.ObjectOutputStream): Unit = {
    writeInternal(new CommonOutputStream {
      override def writeInt(value: Int): Unit = out.writeInt(value)

      override def write(value: Array[Byte]): Unit = out.write(value)

      override def writeString(str: String): Unit = out.writeUTF(str)

      override def writeObject(obj: AnyRef): Unit = out.writeObject(obj)
    })
  }

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    readInternal(new CommonInputStream {
      override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

      override def skip(len: Int): Unit = in.skip(len)

      override def readInt(): Int = in.readInt()

      override def readString(): String = in.readUTF()

      override def readObject(): AnyRef = in.readObject()
    })
  }

  override def read(kryo: Kryo, in: Input): Unit = {
    readInternal(new CommonInputStream {
      override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

      override def skip(len: Int): Unit = in.skip(len)

      override def readInt(): Int = in.readInt()

      override def readString(): String = in.readString()

      override def readObject(): AnyRef = kryo.readClassAndObject(in)
    })
  }

  override def write(kryo: Kryo, out: Output): Unit = {
    writeInternal(new CommonOutputStream {
      override def writeInt(value: Int): Unit = out.writeInt(value)

      override def write(value: Array[Byte]): Unit = out.write(value)

      override def writeString(value: String): Unit = out.writeString(value)

      override def writeObject(obj: AnyRef): Unit = kryo.writeClassAndObject(out, obj)
    })
  }
}

private class RegistryMap[T]() {

  private val logger = LoggerFactory.getLogger(getClass)

  private val registry = new mutable.WeakHashMap[String, T]()

  private def getRegistrySize = registry.size

  def getOrCreate(id: String)(create: => T): (T, Boolean) = {

    val result: Option[T] = registry.get(id)
    result match {
      case Some(value) =>
        logger.debug(s"$id already exists, read from registry. " +
          s"Current registry size: $getRegistrySize")
        return (value, false)
      case _ =>
    }

    registry.synchronized {
      val result: Option[T] = registry.get(id)
      result match {
        case Some(value) =>
          logger.debug(s"$id already exists, read from registry. " +
            s"Current registry size: $getRegistrySize")
          (value, false)
        case _ =>
          logger.debug(s"$id does not exist, created it and added to registry. " +
            s"Current registry size: $getRegistrySize")
          val res = create
          registry.put(id, res)
          (res, true)
      }
    }
  }
}
