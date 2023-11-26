package com.example.myopencl

import android.content.Context
import android.content.res.AssetManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.myopencl.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        initTokenizer(assets)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Example of a call to a native method
        binding.sampleText.text = "good"

        val token = tokenize("a professional photograph of an astronaut riding a horse")
        val tensor = Tensor.fromBlob(token, longArrayOf(1, token.size.toLong()))

        val module = Module.load(assetsFilePath(this, "encoder/text_encoder.ptl"))

        // output shape = (batch_size=1, context_length=77, 1024)
//        val output = module.forward(IValue.from(tensor)).toTensor().dataAsFloatArray

//        val test1 = readFloatFromAssets(this, "encoder/encoder_test1.txt")

        module.destroy()
    }

    /**
     * A native method that is implemented by the 'myopencl' native library,
     * which is packaged with this application.
     */
    external fun initTokenizer(assetManager: AssetManager)
    external fun tokenize(text: String): LongArray

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