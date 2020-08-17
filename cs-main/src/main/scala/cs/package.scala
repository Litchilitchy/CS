
import cs.module.{AbstractModule, Activity}

package object cs {
  type Module[T] = AbstractModule[Activity, Activity, T]
}
