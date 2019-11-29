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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.ccl
import com.intel.analytics.bigdl.ccl.CCLAdapter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import java.util.concurrent.ConcurrentHashMap
import scala.collection.mutable
import scala.reflect.ClassTag

class CCLParameterSynchronizer[T: ClassTag](val parId: Int, val totalPartition: Int)
  (implicit ev: TensorNumeric[T]) extends DistriParameterSynchronizer[T]{

  val k8sAPIServer: String = System.getProperty("ccl.k8sserver")
  require(k8sAPIServer != null, "ccl.k8sserver must be set")
  CCLAdapter.load()
  ccl.CCLAdapter.setEnv(k8sAPIServer, totalPartition)
  ccl.CCLAdapter.doInit()

  private val impl = new BlockManagerParameterSynchronizer[T](parId, totalPartition)
  val comm: ccl.CCLAdapter = new ccl.CCLAdapter(0)

  case class LayerInfo(name: String, globalSize: Int, priority: Int,
    weights: Tensor[T], grads: Tensor[T],
    shadowGrads: Tensor[T], outGrads: Tensor[T], cacheId: Long)

  private val layers = new mutable.HashMap[String, LayerInfo]
  private val requests = new ConcurrentHashMap[String, ccl.CCLAdapter.RequestInfo]

  override def init(name: String, globalSize: Int, priority: Int,
    weights: Tensor[T], grads: Tensor[T]): Unit = {
    val shadow = Tensor(grads.size())
    val cacheId = comm.createTensorCache(name, grads.nElement())
    layers.update(name, LayerInfo(name, globalSize,
      priority, weights, grads, shadow, Tensor(grads.size()), cacheId))
    shadow.copy(grads)
    impl.init(name, globalSize, priority, null, shadow)
  }

  override def put(name: String): Unit = {
    val layer = layers(name)
    val arr = layer.grads.storage().array().asInstanceOf[Array[Float]]
    val offset = layer.grads.storageOffset()
    val len = layer.grads.nElement()
    layer.shadowGrads.copy(layer.grads)

    requests.put(name, comm.allReduceFloatCached(layer.cacheId, arr, offset - 1))
    impl.put(name)
  }

  def arrayDiff(a: Tensor[T], b: Tensor[T]): T = {
    Tensor[T](a.size()).copy(a).sub(b).sumSquare()
  }
  override def get(name: String): (Tensor[T], Tensor[T]) = {
    val layer = layers(name)
    val req = requests.get(name)
    if (req != null) {
      req.await()
      val ret = layer.outGrads
      req.get(ret.storage().array().asInstanceOf[Array[Float]],
        ret.storageOffset() - 1)
      ret.div(ev.fromType(totalPartition))
      requests.remove(name)

      val (wei, gra) = impl.get(name)
      println(name + " PAPAPA3 True Size:" + gra.sum())
      println(name + " PAPAPA3 My result grad:" + ret.sum())
      println(name + " PAPAPA4 DIFF " + arrayDiff(gra, ret))
      (layer.weights, ret)
    } else {
      println("Request not found " + name)
      (null, null)
    }


  }

  override def clear(): Unit = {}

  override def partitionId: Int = parId
}
