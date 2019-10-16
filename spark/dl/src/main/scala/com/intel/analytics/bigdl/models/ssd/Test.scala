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
import com.intel.analytics.bigdl.dataset.segmentation.COCO.{COCODataset, COCOResult}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Transformer}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Module, SpatialShareConvolution}
import com.intel.analytics.bigdl.optim.{MAPUtil, MeanAveragePrecisionObjectDetection, ValidationMethod}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, FeatureTransformer, ImageFeature, ImageFrameToSample, MTImageFeatureToBatch, MatToFloats, PixelBytesToMat, RoiImageInfo}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, RandomTransformer, Resize}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.{RoiLabel, RoiNormalize}
import com.intel.analytics.bigdl.utils.{Engine, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scopt.OptionParser

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  case class SSDParams(
    folder: String = ".",
    model: String = "model",
    partitions: Int = 2,
    batchSize: Int = 2,
    resolution: Int = 512,
    output: Option[String] = None,
    categories: Option[String] = None
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
    opt[String]('o', "output")
      .text("The output path as COCO json format")
      .action((x, c) => c.copy(output = Some(x)))
    opt[String]('c', "categories")
      .text("The category mapping file that converts" +
        " continuous category index into COCO category id")
      .action((x, c) => c.copy(categories = Some(x)))
  }

  def test(rdd: RDD[ImageFeature], model: Module[Float], preprocessor: FeatureTransformer,
    batchSize: Int, resolution: Int, partitions: Int)
  : RDD[ImageFeature] = {
    model.evaluate()
    val bcastModel = ModelBroadcast[Float]().broadcast(rdd.sparkContext, model)
    val toBatch = MTImageFeatureToBatch(resolution, resolution,
      toRGB = false, extractRoi = true, batchSize = partitions,
      transformer = preprocessor)
    val bcastToBatch = rdd.sparkContext.broadcast(toBatch)
    val _batchSize = batchSize
    rdd.mapPartitions(dataIter => {
      val localModel = bcastModel.value()
      val localTransformer = bcastToBatch.value.cloneTransformer()
      dataIter.grouped(_batchSize).flatMap(imfs => {
        val wrappedItr = DataSet.array(imfs.toArray).data(false)
        val miniBatch = localTransformer(wrappedItr)
        imfs.toIterator.zip(miniBatch).map{ case(imf, batch) =>
          val in = batch.getInput()
          val result = localModel.forward(in)
          imf(ImageFeature.predict) = (result, batch.getTarget())
          imf
        }
      })
    })
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
        /*  .filter(imf => {
            if (COCODataset.fileName2ImgId(imf[String](ImageFeature.uri)) < 3000) {
              println(imf[String](ImageFeature.uri))
              true
            } else false
          }) */
      val ds = rawDs
      val model = Module.loadModule[Float](param.model).evaluate()

      val preProcessor =
        PixelBytesToMat() ->
        RoiNormalize() ->
        Resize(param.resolution, param.resolution) ->
        ChannelNormalize(123f, 117f, 104f, 1, 1, 1) ->
        MatToFloats(validHeight = param.resolution, validWidth = param.resolution)
      /* val eval = MeanAveragePrecisionObjectDetection.createPascalVOC(81, useVoc2007 = true,
        topK = 100
      ) */
      val eval = MeanAveragePrecisionObjectDetection.createCOCO(81)
      val outputTarget = test(ds, model, preProcessor,
        param.batchSize, param.resolution, param.partitions)
      if (param.output.isDefined) {
        require(param.batchSize == 1, "If you need to output the result in JSON, the batchSize" +
          " must be 1")
        require(param.categories.isDefined, "If you need to output the result in JSON, the " +
          " category mapping file should be given be -c")
        val cateMapping = Source.fromFile(param.categories.get).getLines.zipWithIndex.map {
          case (line, idx) => (idx + 1L, line.toInt)
        }.toMap
        val results = outputTarget.flatMap{imf =>
          val prediction = imf[(Activity, Activity)](ImageFeature.predict)
          val output = prediction._1.toTensor[Float]
          val target = prediction._2.toTable
          val imgId = COCODataset.fileName2ImgId(imf[String](ImageFeature.uri))
          val result = new ArrayBuffer[COCOResult]()
          MAPUtil.parseSegmentationTensorResult(output,
            (localImgId, label, score, x1, y1, x2, y2) => {
              val x = Math.round(x1 * imf.getOriginalWidth)
              val w = Math.round((x2 - x1) * imf.getOriginalWidth)
              val y = Math.round(y1 * imf.getOriginalHeight)
              val h = Math.round((y2 - y1) * imf.getOriginalHeight)
              result += new COCOResult(imgId, label, Array(x, y, w, h), score)
            })
          result.toIterator
        }.collect().map(r => new COCOResult(r.imageId, cateMapping(r.categoryId), r.bbox, r.score))
        COCODataset.writeResultsToJsonFile(results, param.output.get)
      } else {
        val evalBcast = sc.broadcast(eval)
        val evalResult = outputTarget.map{imf =>
          val prediction = imf[(Activity, Activity)](ImageFeature.predict)
          val output = prediction._1.toTensor[Float]
          val target = prediction._2.toTable
          evalBcast.value(output, target)
        }.reduce((left, right) => {
          left + right
        })
        println(s"${eval} is ${evalResult}")
      }
    }
  }
}
