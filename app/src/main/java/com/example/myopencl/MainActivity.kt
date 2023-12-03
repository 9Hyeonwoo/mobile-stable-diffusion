package com.example.myopencl

import android.content.Context
import android.content.res.AssetManager
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.myopencl.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private lateinit var module: Module

    private val token: LongArray by lazy {
        tokenize("a professional photograph of an astronaut riding a horse")
    }

    private var encodeResult: FloatArray? = null
    private var ptlEncoderResult: FloatArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        initOpenCL(assets)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.encodeButton.setOnClickListener {
            encodeResult = encode(token)
            compareTextEncoder()
        }

        binding.encodePytorchButton.setOnClickListener {
            if (!::module.isInitialized) {
                module = Module.load(mediaFilePath("encoder/text_encoder.ptl"))
            }
            val tensor = Tensor.fromBlob(token, longArrayOf(1, token.size.toLong()))
            // output shape = (batch_size=1, context_length=77, 1024)
            ptlEncoderResult = module.forward(IValue.from(tensor)).toTensor().dataAsFloatArray
            compareTextEncoder()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        module.destroy()
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

        Log.d("__TEST__", "maxDiff: $maxDiff, num: $num, encodeResult[$id]: ${encodeResult[id]}, ptlEncoderResult[$id]: ${ptlEncoderResult[id]}")
    }

    /**
     * A native method that is implemented by the 'myopencl' native library,
     * which is packaged with this application.
     */
    external fun initOpenCL(assetManager: AssetManager)
    external fun tokenize(text: String): LongArray
    external fun encode(token: LongArray): FloatArray
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