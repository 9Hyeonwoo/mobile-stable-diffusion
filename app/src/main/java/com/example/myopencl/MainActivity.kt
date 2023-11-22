package com.example.myopencl

import android.content.res.AssetManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import com.example.myopencl.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        initTokenizer(assets)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Example of a call to a native method
        binding.sampleText.text = "good"

        tokenize("hello world").forEach {
            Log.d("__TEST__token", it.toString())
        }
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
    }
}