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

package com.intel.analytics.bigdl.models.ssd

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.DataSet.{ImageFolder, SeqFileFolder}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Transformer}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.{Module, SpatialShareConvolution}
import com.intel.analytics.bigdl.optim.{MeanAveragePrecisionObjectDetection, ValidationMethod}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, MTImageFeatureToBatch, MatToFloats, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, RandomTransformer, Resize}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiNormalize
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  case class SSDParams(
    folder: String = ".",
    model: String = "model",
    partitions: Int = 2,
    batchSize: Int = 10,
    resolution: Int = 512
  )

  private val parser = new OptionParser[SSDParams]("BigDL SSD test") {
    head("BigDL SSD test")
    opt[String]('f', "folder")
      .text("where you put the sequence files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("where you put the model file")
      .action((x, c) => c.copy(model = x))
    opt[Int]('p', "partitions")
      .text("partition num")
      .action((x, c) => c.copy(partitions = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('r', "resolution")
      .text("resolution, 300 or 512")
      .action((x, c) => c.copy(resolution = x))
  }

  def test(rdd: RDD[ImageFeature], model: Module[Float], preProcessor: Transformer[ImageFeature,
    MiniBatch[Float]], evaluator: ValidationMethod[Float]): Unit = {
    model.evaluate()
    val broadcastModel = ModelBroadcast[Float]().broadcast(rdd.sparkContext, model)
    val broadcastEvaluator = rdd.sparkContext.broadcast(evaluator)
    val broadcastTransformers = rdd.sparkContext.broadcast(preProcessor)
    val output = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localEvaluator = broadcastEvaluator.value.clone()
      val localTransformer = broadcastTransformers.value.cloneTransformer()
      val miniBatch = localTransformer(dataIter)
      miniBatch.map(batch => {
        val in = batch.getInput()
        val result = localModel.forward(in)
        localEvaluator(result, batch.getTarget())
      })
    }).reduce((left, right) => {
      left + right
    })
    println(s"${evaluator} is ${output}")
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, SSDParams()).foreach { param =>
      val conf = Engine.createSparkConf()
        .setAppName("Test SSD")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init
      val rawDs = SeqFileFolder.filesToRoiImageFrame(param.folder, sc, Some(param.partitions))
        .toDistributed().rdd
        .filter(imf => {
          imf[Array[Byte]](ImageFeature.bytes).length >= imf.getOriginalWidth * imf
            .getOriginalHeight * 3
        })
          .filter(imf => {
            if (imf[String](ImageFeature.uri).split("_")(2).substring(0, 12).toInt < 300) {
              println(imf[String](ImageFeature.uri))
              true
            } else false
          })
      val ds = DataSet.rdd(rawDs).data(false)
      val model = Module.loadModule[Float](param.model).evaluate()

      val preProcessor = MTImageFeatureToBatch(param.resolution, param.resolution,
        param.batchSize,
        PixelBytesToMat() ->
        RoiNormalize() ->
        Resize(param.resolution, param.resolution) ->
        ChannelNormalize(123f, 117f, 104f, 1, 1, 1) ->
        MatToFloats(validHeight = param.resolution,
          validWidth = param.resolution), toRGB = false, extractRoi = true)
      val eval = new MeanAveragePrecisionObjectDetection[Float](81,
        useVoc2007 = true, skipClass = 0)
      test(ds, model, preProcessor, eval)
    }
  }
}
