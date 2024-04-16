package com.example.myopencl

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.myopencl.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.util.Random
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding


    private val token: LongArray by lazy {
        tokenize("a professional photograph of an astronaut riding a horse")
    }

    private var initialized = false

    val scope = CoroutineScope(Job() + Dispatchers.IO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.encodeButton.setOnClickListener {
            thread(start = true) {
//            scope.launch {
//                MainScope().launch {
//                    binding.progressBar.visibility = View.VISIBLE
//                }
                if (!initialized) {
                    // about 1m 40s
                    Log.d("__TEST__", "start initOpenCL")
                    initOpenCL(assets)
                    Log.d("__TEST__", "end initOpenCL")
//                    val result = sample(FloatArray(77 * 1024))
                    /**
                     * decode() block
                    val result = decode()
                    MainScope().launch {
                        drawImage(result)
                    }
                     */
//                    val result = sample(FloatArray(77 * 1024))
                    initialized = true
                }
                /**
                 * encode() block
                 */
                val encodeResult = encode(token)
                destroyOpenCL()
                initialized = false
//                MainScope().launch {
//                    binding.progressBar.visibility = View.GONE
//                }
            }
        }

        binding.encodePytorchButton.setOnClickListener {
            initOpenCL(assets)
            val token = token
            val condition: FloatArray
            val start = System.currentTimeMillis()
            run {
                // about 8s
                Log.d("__TEST__", "start torchEncoder")
                val torchEncoder = checkTime("encode_init") {
                    Module.load(mediaFilePath("encoder/text_encoder.ptl"))
                }

                val tensor = Tensor.fromBlob(token, longArrayOf(1, token.size.toLong()))
                Log.d("__TEST__", "start encode")
                // output shape = (batch_size=1, context_length=77, 1024)

                condition = checkTime("encode_inference") {
                    torchEncoder.forward(IValue.from(tensor)).toTensor().dataAsFloatArray
                }
                torchEncoder.destroy()
                Log.d("__TEST__", "end torchEncoder")
            }
            return@setOnClickListener
            val latent: FloatArray
            run {
                val random = Random(45)
                // val condition = FloatArray(77 * 1024) { random.nextGaussian().toFloat() }
                // about 21s
                Log.d("__TEST__", "start torch UNet")
                val torchUNetInput = Module.load(mediaFilePath("unet/unet_input.ptl"))

                Log.d("__TEST__", "start unet_input")
                val x = FloatArray(4 * 64 * 64) { random.nextGaussian().toFloat() }
                val hsInput = unet(torchUNetInput, x, longArrayOf(1, 4, 64, 64), 981, condition)
                Log.d("__TEST__", "end unet_input")
                torchUNetInput.destroy()

                val torchUNetMid = Module.load(mediaFilePath("unet/unet_mid.ptl"))
                Log.d("__TEST__", "start unet_mid")
                val hsMid =
                    unet(torchUNetMid, hsInput, longArrayOf(hsInput.size.toLong()), 1, condition)
                torchUNetMid.destroy()
                Log.d("__TEST__", "start unet_mid")

                //  val hs_mid = NpyFile.read(Path(mediaFilePath("unet/unet_mid_result.npy"))).asFloatArray()
                val torchUNetOutput0_3 = Module.load(mediaFilePath("unet/unet_out0_3.ptl"))
                val hs_out_0_3 = unet(
                    torchUNetOutput0_3,
                    hsMid,
                    longArrayOf(hsMid.size.toLong()),
                    981,
                    condition
                )
                Log.d("__TEST__", "end unet_out_0_3")
                torchUNetOutput0_3.destroy()

                //  val hs_out_0_3 = FloatArray(6389760) { random.nextGaussian().toFloat() }
                val torchUNetOutput4_11 = Module.load(mediaFilePath("unet/unet_out4_11.ptl"))
                latent = unet(
                    torchUNetOutput4_11,
                    hs_out_0_3,
                    longArrayOf(hs_out_0_3.size.toLong()),
                    981,
                    condition
                )
                Log.d("__TEST__", "end unet_out_4_11")
                torchUNetOutput4_11.destroy()
                Log.d("__TEST__", "end torch UNet")
            }
            val img: FloatArray
            run {
                // val latent =
                //    NpyFile.read(Path(mediaFilePath("decoder/test/test_seed_45_step_50_sample.npy")))
                Log.d("__TEST__", "start torchDecoder")
                val torchDecoder = Module.load(mediaFilePath("decoder/decoder.ptl"))
                Log.d("__TEST__", "end torchDecoder")


                Log.d("__TEST__", "start decode")
                val procLatent = latent.map { x -> 1f / 0.18215f * x }.toFloatArray()
                val tensor = IValue.from(
                    Tensor.fromBlob(
                        procLatent,
                        longArrayOf(1, 4, 64, 64)
                    )
                )
                img = torchDecoder.forward(tensor).toTensor().dataAsFloatArray
                Log.d("__TEST__", "end decode")
                torchDecoder.destroy()
            }
            val stop = System.currentTimeMillis()
            Log.d("__TEST__", "torch unet time: ${stop - start} ms")
            drawImage(img)
        }
    }

    private fun drawImage(imageArray: FloatArray) {
        val height = 512
        val width = 512
        val imgByte = imageArray.map { x ->
            (((x + 1f) / 2f).coerceIn(0f, 1f) * 255f).toUInt().toUByte()
        }

        val pixels = IntArray(height * width)
        for (i in 0 until height) {
            for (j in 0 until width) {
                val red = imgByte[i * width + j].toInt() and 0xFF
                val green = imgByte[height * width + i * width + j].toInt() and 0xFF
                val blue = imgByte[2 * height * width + i * width + j].toInt() and 0xFF
                Color.rgb(red, green, blue).also {
                    pixels[i * width + j] = it
                }
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        binding.imageView.setImageBitmap(bitmap)
    }

    override fun onDestroy() {
        super.onDestroy()

        destroyOpenCL()
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

    private inline fun<T> checkTime(tag: String, block: () -> T): T {
        val start = System.currentTimeMillis()
        val result = block()
        val stop = System.currentTimeMillis()
        Log.d("__TEST__", "$tag time: ${stop - start} ms")
        return result
    }

    /**
     * A native method that is implemented by the 'myopencl' native library,
     * which is packaged with this application.
     */
    external fun initOpenCL(assetManager: AssetManager)
    external fun tokenize(text: String): LongArray
    external fun encode(token: LongArray): FloatArray
    external fun sample(condition: FloatArray): FloatArray
    external fun decode(): FloatArray
    external fun destroyOpenCL()

    companion object {
        // Used to load the 'myopencl' library on application startup.
        init {
            System.loadLibrary("myopencl")
        }

        private fun mediaFilePath(name: String): String {
            return Environment.getExternalStorageDirectory().absolutePath + "/Android/media/com.example.myopencl/$name"
        }
    }
}