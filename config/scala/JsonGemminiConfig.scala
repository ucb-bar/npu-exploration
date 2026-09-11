// ============================================================================
// The Verilator side of the SAME recipe the golden model and the spike build read
// (config/recipes/*.json, config/recipe.py). Resolves GEMMINI_RECIPE_JSON into a
// GemminiArrayConfig, so one file defines the hardware for all three models.
//
// Install into a chipyard tree at:
//     generators/chipyard/src/main/scala/config/JsonGemminiConfig.scala
// then:  make CONFIG=JsonGemminiConfig GEMMINI_RECIPE_JSON=<abs path to recipe>
//
// STATUS: written, NEVER ELABORATED. No Verilator simulator has been built in this
// tree at all -- sims/verilator/generated-src holds elaboration output only, for the
// fixed VendorStandaloneMxRocketConfig. Treat everything below as unexercised.
//
// KNOWN ISSUE, fix before trusting a recipe build: the base preset below is
// testMxFPConfig, but the config the bring-up actually elaborates (and that the
// emitted C's ABI assumes) is standaloneMxFPConfig. The arithmetic is identical --
// no MX config overrides the precision lists -- but the PLUMBING is not:
// standaloneMxFPConfig sets scale_mem.baseAddr = 0x20000000 (ConfigsFP.scala:342),
// which is the address gemmini_mx_rocket.h hardcodes. A recipe built on this base
// would put the scale memory at 0x10008000 while the driver writes scales to
// 0x20000000. Deliberately left unchanged rather than fixed blind: it cannot be
// tested without a Verilator build.
//
// The Python side ignores the `software` and `runtime` sections; this file rejects
// unknown top-level keys, so they are listed in ALLOWED_TOP below and dropped.
// ============================================================================

package chipyard

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.diplomacy.{LazyModule, ValName}
import freechips.rocketchip.tile.BuildRoCC
import gemmini.{Gemmini, GemminiArrayConfig, GemminiMxFPConfigs, MxFloat, GemminiLUTConfig, Float => GmFloat}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse

import scala.io.Source

// Reads a recipe JSON (path given by the GEMMINI_RECIPE_JSON env var) and
// resolves it into a GemminiArrayConfig, starting from one of the two
// known-good base presets already used together in ConfigsFP.scala
// (testMxFPConfig / testRequantizerLutMxFPConfig), overriding only the
// fields present in the JSON. This is the only place a recipe's JSON
// shape (see recipes/FIELDS.md) is interpreted on the Scala side.
object JsonGemminiRecipe {

  type ArrCfg = GemminiArrayConfig[MxFloat, GmFloat, GmFloat]

  private val ALLOWED_TOP     = Set("name", "description", "array", "types", "mx", "accumulator",
                            "supported_backends", "software", "runtime", "provenance")
  // "description"/"software"/"runtime"/"provenance" are consumed by config/recipe.py only:
  // they change no gate, so they are accepted here and ignored.
  private val ALLOWED_ARRAY   = Set("meshRows", "meshColumns", "tileRows", "tileColumns")
  private val ALLOWED_TYPES   = Set(
    "inputType", "weightType", "accType",
    "inputTypeProjected", "weightTypeProjected", "accTypeProjected",
    "spatialArrayInputType", "spatialArrayWeightType", "spatialArrayOutputType",
    "meshProdPrecisionList", "meshAccPrecisionList"
  )
  private val ALLOWED_MX       = Set("scaleSize", "enable_lut", "lut")
  private val ALLOWED_ACC      = Set("acc_read_full_width", "acc_read_small_width")
  private val ALLOWED_MXFLOAT  = Set("expWidth", "sigWidth", "count", "isRecoded", "pad")
  private val ALLOWED_LUT      = Set("numBits", "numEntries", "rdataWidth", "raddrWidth", "lutUpdateRegularityWidth")

  class RecipeError(msg: String) extends RuntimeException(msg)

  def envPath: String =
    sys.env.getOrElse("GEMMINI_RECIPE_JSON",
      throw new RecipeError("GEMMINI_RECIPE_JSON env var not set"))

  def load(): ArrCfg = resolve(parse(Source.fromFile(envPath).mkString))

  private def checkKeys(fields: Map[String, JValue], allowed: Set[String], where: String): Unit = {
    val extra = fields.keySet -- allowed
    if (extra.nonEmpty)
      throw new RecipeError(s"Unrecognized recipe key(s) in $where: ${extra.mkString(", ")}. Allowed: ${allowed.mkString(", ")}")
  }

  private def asObjMap(v: JValue, where: String): Map[String, JValue] = v match {
    case JObject(fs) => fs.toMap
    case other        => throw new RecipeError(s"$where: expected an object, got $other")
  }

  private def asInt(v: JValue, where: String): Int = v match {
    case JInt(n)    => n.toInt
    case JDouble(d) => d.toInt
    case JLong(n)   => n.toInt
    case other      => throw new RecipeError(s"$where: expected an int, got $other")
  }

  private def asBool(v: JValue, where: String): Boolean = v match {
    case JBool(b) => b
    case other    => throw new RecipeError(s"$where: expected a bool, got $other")
  }

  private def asIntSeq(v: JValue, where: String): Seq[Int] = v match {
    case JArray(items) => items.zipWithIndex.map { case (x, i) => asInt(x, s"$where[$i]") }
    case other          => throw new RecipeError(s"$where: expected an array of ints, got $other")
  }

  private def parseMxFloat(v: JValue, where: String): MxFloat = {
    val f = asObjMap(v, where)
    checkKeys(f, ALLOWED_MXFLOAT, where)
    val expWidth = f.get("expWidth").map(asInt(_, s"$where.expWidth"))
      .getOrElse(throw new RecipeError(s"$where.expWidth is required"))
    val sigWidth = f.get("sigWidth").map(asInt(_, s"$where.sigWidth"))
      .getOrElse(throw new RecipeError(s"$where.sigWidth is required"))
    val count = f.get("count").map(asInt(_, s"$where.count"))
      .getOrElse(throw new RecipeError(s"$where.count is required"))
    val isRecoded = f.get("isRecoded").map(asBool(_, s"$where.isRecoded")).getOrElse(false)
    val pad = f.get("pad").map(asBool(_, s"$where.pad")).getOrElse(true)
    MxFloat(expWidth, sigWidth, count, isRecoded, pad)
  }

  private def parseMxFloatList(v: JValue, where: String): Seq[MxFloat] = v match {
    case JArray(items) => items.zipWithIndex.map { case (item, i) => parseMxFloat(item, s"$where[$i]") }
    case other           => throw new RecipeError(s"$where: expected an array, got $other")
  }

  private def parseLutConfig(v: JValue, where: String): GemminiLUTConfig = {
    val f = asObjMap(v, where)
    checkKeys(f, ALLOWED_LUT, where)
    var lut = GemminiLUTConfig()
    f.get("numBits").foreach(x => lut = lut.copy(numBits = asIntSeq(x, s"$where.numBits")))
    f.get("numEntries").foreach(x => lut = lut.copy(numEntries = asIntSeq(x, s"$where.numEntries")))
    f.get("rdataWidth").foreach(x => lut = lut.copy(rdataWidth = asInt(x, s"$where.rdataWidth")))
    f.get("raddrWidth").foreach(x => lut = lut.copy(raddrWidth = asInt(x, s"$where.raddrWidth")))
    f.get("lutUpdateRegularityWidth").foreach(x =>
      lut = lut.copy(lutUpdateRegularityWidth = asInt(x, s"$where.lutUpdateRegularityWidth")))
    lut
  }

  def resolve(json: JValue): ArrCfg = {
    val root = asObjMap(json, "recipe root")
    checkKeys(root, ALLOWED_TOP, "recipe root")

    val mxFields = root.get("mx").map(v => asObjMap(v, "mx")).getOrElse(Map.empty)
    if (root.contains("mx")) checkKeys(mxFields, ALLOWED_MX, "mx")

    val hasLut = mxFields.get("lut").exists {
      case JNull => false
      case _     => true
    }

    var cfg: ArrCfg =
      if (hasLut) GemminiMxFPConfigs.testRequantizerLutMxFPConfig
      else GemminiMxFPConfigs.testMxFPConfig

    val arrayFields = root.get("array").map(v => asObjMap(v, "array")).getOrElse(Map.empty)
    if (root.contains("array")) checkKeys(arrayFields, ALLOWED_ARRAY, "array")

    val typesFields = root.get("types").map(v => asObjMap(v, "types")).getOrElse(Map.empty)
    if (root.contains("types")) checkKeys(typesFields, ALLOWED_TYPES, "types")

    val accFields = root.get("accumulator").map(v => asObjMap(v, "accumulator")).getOrElse(Map.empty)
    if (root.contains("accumulator")) checkKeys(accFields, ALLOWED_ACC, "accumulator")

    // Compute every override against the untouched base first, then apply
    // them all in ONE .copy() call below. GemminiArrayConfig's constructor
    // validates cross-field invariants (e.g. BLOCK_ROWS == BLOCK_COLS) on
    // every .copy(), so applying fields one at a time can transiently
    // construct an invalid intermediate state (new meshRows, still-default
    // meshColumns) even when the final combination is valid.
    val meshRows    = arrayFields.get("meshRows").map(asInt(_, "array.meshRows")).getOrElse(cfg.meshRows)
    val meshColumns = arrayFields.get("meshColumns").map(asInt(_, "array.meshColumns")).getOrElse(cfg.meshColumns)
    val tileRows    = arrayFields.get("tileRows").map(asInt(_, "array.tileRows")).getOrElse(cfg.tileRows)
    val tileColumns = arrayFields.get("tileColumns").map(asInt(_, "array.tileColumns")).getOrElse(cfg.tileColumns)

    val inputType             = typesFields.get("inputType").map(parseMxFloat(_, "types.inputType")).getOrElse(cfg.inputType)
    val weightType            = typesFields.get("weightType").map(parseMxFloat(_, "types.weightType")).getOrElse(cfg.weightType)
    val accType               = typesFields.get("accType").map(parseMxFloat(_, "types.accType")).getOrElse(cfg.accType)
    val inputTypeProjected    = typesFields.get("inputTypeProjected").map(parseMxFloat(_, "types.inputTypeProjected")).getOrElse(cfg.inputTypeProjected)
    val weightTypeProjected   = typesFields.get("weightTypeProjected").map(parseMxFloat(_, "types.weightTypeProjected")).getOrElse(cfg.weightTypeProjected)
    val accTypeProjected      = typesFields.get("accTypeProjected").map(parseMxFloat(_, "types.accTypeProjected")).getOrElse(cfg.accTypeProjected)
    val spatialArrayInputType  = typesFields.get("spatialArrayInputType").map(parseMxFloat(_, "types.spatialArrayInputType")).getOrElse(cfg.spatialArrayInputType)
    val spatialArrayWeightType = typesFields.get("spatialArrayWeightType").map(parseMxFloat(_, "types.spatialArrayWeightType")).getOrElse(cfg.spatialArrayWeightType)
    val spatialArrayOutputType = typesFields.get("spatialArrayOutputType").map(parseMxFloat(_, "types.spatialArrayOutputType")).getOrElse(cfg.spatialArrayOutputType)
    val meshProdPrecisionList = typesFields.get("meshProdPrecisionList").map(parseMxFloatList(_, "types.meshProdPrecisionList")).getOrElse(cfg.meshProdPrecisionList)
    val meshAccPrecisionList  = typesFields.get("meshAccPrecisionList").map(parseMxFloatList(_, "types.meshAccPrecisionList")).getOrElse(cfg.meshAccPrecisionList)

    val scaleSize  = mxFields.get("scaleSize").map(asInt(_, "mx.scaleSize")).getOrElse(cfg.scaleSize)
    val enableLut  = mxFields.get("enable_lut").map(asBool(_, "mx.enable_lut")).getOrElse(cfg.enable_lut)
    val lutOpt = mxFields.get("lut") match {
      case Some(JNull) | None => cfg.lut
      case Some(x)            => Some(parseLutConfig(x, "mx.lut"))
    }

    val accReadFullWidth  = accFields.get("acc_read_full_width").map(asBool(_, "accumulator.acc_read_full_width")).getOrElse(cfg.acc_read_full_width)
    val accReadSmallWidth = accFields.get("acc_read_small_width").map(asBool(_, "accumulator.acc_read_small_width")).getOrElse(cfg.acc_read_small_width)

    cfg = cfg.copy(
      meshRows = meshRows, meshColumns = meshColumns, tileRows = tileRows, tileColumns = tileColumns,
      inputType = inputType, weightType = weightType, accType = accType,
      inputTypeProjected = inputTypeProjected, weightTypeProjected = weightTypeProjected, accTypeProjected = accTypeProjected,
      spatialArrayInputType = spatialArrayInputType, spatialArrayWeightType = spatialArrayWeightType, spatialArrayOutputType = spatialArrayOutputType,
      meshProdPrecisionList = meshProdPrecisionList, meshAccPrecisionList = meshAccPrecisionList,
      scaleSize = scaleSize, enable_lut = enableLut, lut = lutOpt,
      acc_read_full_width = accReadFullWidth, acc_read_small_width = accReadSmallWidth
    )

    if (cfg.meshProdPrecisionList.length != cfg.meshColumns)
      throw new RecipeError(
        s"types.meshProdPrecisionList length (${cfg.meshProdPrecisionList.length}) must equal meshColumns (${cfg.meshColumns})")
    if (cfg.meshAccPrecisionList.length != cfg.meshColumns)
      throw new RecipeError(
        s"types.meshAccPrecisionList length (${cfg.meshAccPrecisionList.length}) must equal meshColumns (${cfg.meshColumns})")

    cfg
  }
}

class WithJsonGemmini extends Config((site, here, up) => {
  case BuildRoCC => Seq(
    (p: Parameters) => {
      implicit val q = p
      implicit val v = implicitly[ValName]
      LazyModule(new Gemmini(JsonGemminiRecipe.load()))
    }
  )
})

class JsonGemminiConfig extends Config(
  new WithJsonGemmini ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
