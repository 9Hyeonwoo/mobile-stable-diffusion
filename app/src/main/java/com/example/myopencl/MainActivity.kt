package com.example.myopencl

import android.content.Context
import android.content.res.AssetManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.myopencl.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.util.Random

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding


    private val token: LongArray by lazy {
        tokenize("a professional photograph of an astronaut riding a horse")
    }

    private var encodeResult: FloatArray? = null
    private var ptlEncoderResult: FloatArray? = null

    private var initialized = false

    private var torchUNet: Module? = null

    val scope = CoroutineScope(Job() + Dispatchers.IO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.encodeButton.setOnClickListener {
            scope.launch {
                if (!initialized) {
                    // about 1m 40s
                    Log.d("__TEST__", "start initOpenCL")
                    initOpenCL(assets)
                    Log.d("__TEST__", "end initOpenCL")
                    initialized = true
                }
                encodeResult = encode(token)
                destroyOpenCL()
                initialized = false
            }
        }

        binding.encodePytorchButton.setOnClickListener {
             run {
                // about 8s
                Log.d("__TEST__", "start torchEncoder")
                val torchEncoder = Module.load(mediaFilePath("encoder/text_encoder.ptl"))
                Log.d("__TEST__", "end torchEncoder")

                 val tensor = Tensor.fromBlob(token, longArrayOf(1, token.size.toLong()))
                 Log.d("__TEST__", "start encode")
                 // output shape = (batch_size=1, context_length=77, 1024)
                 ptlEncoderResult = torchEncoder.forward(IValue.from(tensor)).toTensor().dataAsFloatArray
                 Log.d("__TEST__", "end encode")
                 torchEncoder.destroy()
            }
            val latent: FloatArray
            run {
                // about 21s
                Log.d("__TEST__", "start torchUNetInput")
                val torchUNetInput = Module.load(mediaFilePath("unet/unet_input.ptl"))
                Log.d("__TEST__", "end torchUNetInput")

                Log.d("__TEST__", "start unet_input")
//                latent = sample(ptlEncoderResult!!)
                val random = Random(45)
                val x = FloatArray(4 * 64 * 64) { random.nextGaussian().toFloat() }
                val condition = FloatArray(77 * 1024) { random.nextGaussian().toFloat() }
                val hs_input = unet(torchUNetInput, x, longArrayOf(1, 4, 64, 64), 1, condition)
                Log.d("__TEST__", "end unet")
                torchUNetInput.destroy()

                val torchUNetMid = Module.load(mediaFilePath("unet/unet_mid.ptl"))
                Log.d("__TEST__", "start unet_mid")

                val hs_mid = unet(torchUNetMid, hs_input, longArrayOf(1), 1, condition)
                Log.d("__TEST__", "end unet_mid")
                torchUNetMid.destroy()

                val torchUNetOutput = Module.load(mediaFilePath("unet/unet_out.ptl"))
                Log.d("__TEST__", "start unet_out")
                latent = unet(torchUNetOutput, hs_mid, longArrayOf(1), 1, condition)
                Log.d("__TEST__", "end unet_out")
                torchUNetOutput.destroy()
            }
            run {
                Log.d("__TEST__", "start torchDecoder")
                val torchDecoder = Module.load(mediaFilePath("decoder/decoder.ptl"))
                Log.d("__TEST__", "end torchDecoder")

                Log.d("__TEST__", "start decode")
                val procLatent = latent.map { x -> 1f / 0.18215f * x }.toFloatArray()
                val img = torchDecoder.forward(
                    IValue.from(
                        Tensor.fromBlob(
                            procLatent,
                            longArrayOf(1, 4, 64, 64)
                        )
                    )
                ).toTensor().dataAsFloatArray
                val imgByte =
                    img.map { x -> (((x + 1f) / 2f).coerceIn(0f, 1f) * 255f).toUInt().toByte() }
                Log.d("__TEST__", "end decode")

                BitmapFactory.decodeByteArray(imgByte.toByteArray(), 0, imgByte.size).also {
                    binding.imageView.setImageBitmap(it)
                }
                torchDecoder.destroy()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        destroyOpenCL()
    }

    private fun compareTextEncoder() {
        val encodeResult = encodeResult
        val ptlEncoderResult = ptlEncoderResult
        if (encodeResult == null || ptlEncoderResult == null) {
            Log.d("__TEST__", "result is null")
            return
        }

        var maxDiff = 0f
        var num = 0
        var id = 0
        encodeResult.forEachIndexed { index, r1 ->
            val r2 = ptlEncoderResult[index]
            if (r1 != r2) {
                num++
                val diff = Math.abs(r1 - r2)
                if (diff > maxDiff) {
                    maxDiff = diff
                    id = index
                }
            }
        }

        Log.d(
            "__TEST__",
            "maxDiff: $maxDiff, num: $num, encodeResult[$id]: ${encodeResult[id]}, ptlEncoderResult[$id]: ${ptlEncoderResult[id]}"
        )
    }

    fun unet(
        module: Module,
        input: FloatArray,
        inputShape: LongArray,
        step: Long,
        condition: FloatArray
    ): FloatArray {
        val tensorInput = Tensor.fromBlob(input, inputShape)
        val tensorStep = Tensor.fromBlob(longArrayOf(step), longArrayOf(1))
        val tensorCondition = Tensor.fromBlob(condition, longArrayOf(1, 77, 1024))
        return module.forward(
            IValue.from(tensorInput),
            IValue.from(tensorStep),
            IValue.from(tensorCondition)
        ).toTensor().dataAsFloatArray
    }

    /**
     * A native method that is implemented by the 'myopencl' native library,
     * which is packaged with this application.
     */
    external fun initOpenCL(assetManager: AssetManager)
    external fun tokenize(text: String): LongArray
    external fun encode(token: LongArray): FloatArray
    external fun sample(condition: FloatArray): FloatArray
    external fun destroyOpenCL()

    companion object {
        // Used to load the 'myopencl' library on application startup.
        init {
            System.loadLibrary("myopencl")
        }

        fun assetsFilePath(context: Context, assetName: String): String {
            val file = context.filesDir.resolve(assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName).use { inputStream ->
                file.outputStream().use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int = inputStream.read(buffer)
                    while (read != -1) {
                        outputStream.write(buffer, 0, read)
                        read = inputStream.read(buffer)
                    }
                    outputStream.flush()
                }
                return file.absolutePath
            }
        }

        fun mediaFilePath(name: String): String {
            return Environment.getExternalStorageDirectory().absolutePath + "/Android/media/com.example.myopencl/$name"
        }

        fun readFloatFromAssets(context: Context, assetName: String): Sequence<Float> {
            context.assets.open(assetName).use { inputStream ->
                val regex = Regex("-?\\d+.\\d+(e[-|+]\\d+)?")
                return regex.findAll(inputStream.reader().readText()).map { it.value.toFloat() }
            }
        }
    }
}