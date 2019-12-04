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

import com.intel.analytics.bigdl.ccl.CCLAdapter
import com.intel.analytics.bigdl.parameters.FP16CompressedTensor
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import scala.collection.mutable
import scala.reflect.ClassTag

class CCLParameterSynchronizer[T: ClassTag](val parId: Int, val totalPartition: Int)
  (implicit ev: TensorNumeric[T]) extends DistriParameterSynchronizer[T]{

  val k8sAPIServer: String = System.getProperty("ccl.k8sserver")
  require(k8sAPIServer != null, "ccl.k8sserver must be set")
  CCLAdapter.load()
  CCLAdapter.setEnv(k8sAPIServer, totalPartition)
  CCLAdapter.doInit()

  val comm: CCLAdapter = new CCLAdapter(0)

  case class LayerInfo(name: String, globalSize: Int, priority: Int,
    weights: Tensor[T], grads: Tensor[T], cacheId: Long)

  case class RequestInfoWraper(var request: CCLAdapter.RequestInfo, var cnt: Int = 0)

  private val layers = new mutable.HashMap[String, LayerInfo]
  private val requests = new mutable.HashMap[String, RequestInfoWraper]

  override def init(name: String, globalSize: Int, priority: Int,
    weights: Tensor[T], grads: Tensor[T]): Unit = {
    val cacheId = comm.createTensorCache(name, grads.nElement())
    layers.update(name, LayerInfo(name, globalSize,
      priority, weights, grads, cacheId))
    requests.update(name, RequestInfoWraper(null))
  }

  override def put(name: String): Unit = {
    val layer = layers(name)
    val arr = layer.grads.storage().array().asInstanceOf[Array[Float]]
    val offset = layer.grads.storageOffset()
    require(requests.contains(name), s"The layer $name is not in the allreduce request cache")
    val req = requests(name)
    req.cnt += 1
    require(req.request == null, s"There is an outstanding allreduce request for layer $name")
    req.request = comm.allReduceFloatCached(layer.cacheId, arr, offset - 1)
  }


  override def get(name: String): (Tensor[T], Tensor[T]) = {
    val layer = layers(name)
    require(requests.contains(name), s"The layer $name is not in the allreduce request cache")
    val reqWrapper = requests(name)
    if (reqWrapper.cnt == 0) {
      return (null, null)
    }
    val req = reqWrapper.request
    require(req != null)
    req.await()
    val ret = layer.grads
    req.get(ret.storage().array().asInstanceOf[Array[Float]],
      ret.storageOffset() - 1)
    ret.div(ev.fromType(totalPartition))
    reqWrapper.request = null

    val fp16paramAggregated = new FP16CompressedTensor[T](ret.nElement())
    fp16paramAggregated.compress(0, ret, 0, ret.nElement())
    fp16paramAggregated.deCompress(ret)
    (layer.weights, ret)
  }

  override def clear(): Unit = {
    comm.release()
  }

  override def partitionId: Int = parId
}
