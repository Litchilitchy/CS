package module

import scala.reflect.ClassTag

abstract class AbstractNode[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag] {
  var input: B = Tensor[T]()
  var output: B = Activity.allocate[B, T]()
  /**
   * Takes an input object, and computes the corresponding output of the module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  final def forward(input: A): B = {
    try {
      updateOutput(input)
    } catch {
      case e: Exception =>
        throw new Error(e)
    }
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
}
