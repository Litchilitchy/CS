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

package serving.preprocessing

import module.Activity
import com.intel.analytics.bigdl.opencv.OpenCV
import module.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import module.tensor.T
import org.apache.log4j.Logger
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import serving.http.Instances
import serving.utils.SerParams

import scala.collection.mutable.ArrayBuffer

class PreProcessing(param: SerParams) {
  val logger = Logger.getLogger(getClass)

  var tensorBuffer: Array[Tensor[Float]] = null
  var arrayBuffer: Array[Array[Float]] = null

  var byteBuffer: Array[Byte] = null


  def decodeArrowBase64(s: String): Activity = {
    byteBuffer = java.util.Base64.getDecoder.decode(s)
    val instance = Instances.fromArrow(byteBuffer)

    val kvMap = instance.instances.flatMap(insMap => {
      val oneInsMap = insMap.map(kv =>
        if (kv._2.isInstanceOf[String]) {
          if (kv._2.asInstanceOf[String].contains("|")) {
            (kv._1, decodeString(kv._2.asInstanceOf[String]))
          }
          else {
            (kv._1, decodeImage(kv._2.asInstanceOf[String]))
          }
        }
        else {
          (kv._1, decodeTensor(kv._2.asInstanceOf[(
            ArrayBuffer[Int], ArrayBuffer[Float], ArrayBuffer[Int], ArrayBuffer[Int])]))
        }
      ).toList
//      Seq(T(oneInsMap.head, oneInsMap.tail: _*))
      val arr = oneInsMap.map(x => x._2)
      Seq(T.array(arr.toArray))
    })
    kvMap.head
  }
  def decodeString(s: String): Tensor[String] = {
    val eleList = s.split("\\|")
    val tensor = Tensor[String](eleList.length)
    (1 to eleList.length).foreach(i => {
      tensor.setValue(i, eleList(i - 1))
    })
    tensor
  }

  def decodeImage(s: String, idx: Int = 0): Tensor[Float] = {
    byteBuffer = java.util.Base64.getDecoder.decode(s)
    val mat = OpenCVMethod.fromImageBytes(byteBuffer, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
//    Imgproc.resize(mat, mat, new Size(224, 224))
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

    val arrayBuffer = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, arrayBuffer)

    val imageTensor = Tensor[Float](arrayBuffer, Array(height, width, channel))
    if (param.chwFlag) {
      imageTensor.transpose(1, 3)
        .transpose(2, 3).contiguous()
    } else {
      imageTensor
    }
  }
  def decodeTensor(info: (ArrayBuffer[Int], ArrayBuffer[Float],
    ArrayBuffer[Int], ArrayBuffer[Int])): Tensor[Float] = {
    val data = info._2.toArray
    val shape = info._1.toArray
    if (info._3.size == 0) {
      Tensor[Float](data, shape)
    } else {
      val indiceData = info._4.toArray
      val indiceShape = info._3.toArray
      var indice = new Array[Array[Int]](0)
      val colLength = indiceShape(1)
      var arr: Array[Int] = null
      (0 until indiceData.length).foreach(i => {
        if (i % colLength == 0) {
          arr = new Array[Int](colLength)
        }
        arr(i % colLength) = indiceData(i)
        if ((i + 1) % colLength == 0) {
          indice = indice :+ arr
        }
      })
      Tensor.sparse(indice, data, shape)
    }

  }

}
object OpenCVMethod {
  OpenCV.isOpenCVLoaded

  /**
   * convert image file in bytes to opencv mat with BGR
   *
   * @param fileContent bytes from an image file
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   * @return opencv mat
   */
  def fromImageBytes(fileContent: Array[Byte],
                     imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): OpenCVMat = {
    var mat: Mat = null
    var matOfByte: MatOfByte = null
    var result: OpenCVMat = null
    try {
      matOfByte = new MatOfByte(fileContent: _*)
      mat = Imgcodecs.imdecode(matOfByte, imageCodec)
      result = new OpenCVMat(mat)
    } catch {
      case e: Exception =>
        if (null != result) result.release()
        throw e
    } finally {
      if (null != mat) mat.release()
      if (null != matOfByte) matOfByte.release()
    }
    result
  }
}

