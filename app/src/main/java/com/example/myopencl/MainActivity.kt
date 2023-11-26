package com.example.myopencl

import android.content.Context
import android.content.res.AssetManager
import android.os.Bundle
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        initOpenCL(assets)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.encodeButton.setOnClickListener {
            encode(token)
            Log.d(
                "__TEST__",
                "encode[${token[0]}, ${token[1]}, ${token[2]}, ${token[3]}, ${token[4]}]"
            )
        }

        binding.encodePytorchButton.setOnClickListener {

            val tensor = Tensor.fromBlob(token, longArrayOf(1, token.size.toLong()))
            // output shape = (batch_size=1, context_length=77, 1024)
            val output = module.forward(IValue.from(tensor)).toTensor().dataAsFloatArray
            Log.d(
                "__TEST__",
                "output[${output[0]}, ${output[1]}, ${output[2]}, ${output[3]}, ${output[4]}]"
            )
        }


    }

    override fun onDestroy() {
        super.onDestroy()

        module.destroy()
        destroyOpenCL()
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

        fun readFloatFromAssets(context: Context, assetName: String): Sequence<Float> {
            context.assets.open(assetName).use { inputStream ->
                val regex = Regex("-?\\d+.\\d+(e[-|+]\\d+)?")
                return regex.findAll(inputStream.reader().readText()).map { it.value.toFloat() }
            }
        }
    }
}