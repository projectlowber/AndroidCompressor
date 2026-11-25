package com.threep.compressor

import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private enum class Mode { HOME, READY, BUSY }

    private var mode = Mode.HOME
    private var selectedUri: Uri? = null
    private var selectedName: String = ""
    private var selectedSize: Long = 0L
    private var isExtract: Boolean = false

    private lateinit var tvSelectedLabel: TextView
    private lateinit var tvFileName: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvBefore: TextView
    private lateinit var tvAfter: TextView
    private lateinit var tvSaved: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnAction: Button

    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (_: Exception) {}
            onFileSelected(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvSelectedLabel = findViewById(R.id.tvSelectedLabel)
        tvFileName = findViewById(R.id.tvFileName)
        tvStatus = findViewById(R.id.tvStatus)
        tvBefore = findViewById(R.id.tvBefore)
        tvAfter = findViewById(R.id.tvAfter)
        tvSaved = findViewById(R.id.tvSaved)
        progressBar = findViewById(R.id.progressBar)
        btnAction = findViewById(R.id.btnAction)

        btnAction.setOnClickListener {
            when (mode) {
                Mode.HOME -> pickFile.launch(arrayOf("*/*"))
                Mode.READY -> startConversion()
                Mode.BUSY -> {
                    NativeEngine.cancelJob()
                    tvStatus.text = "Cancelling..."
                }
            }
        }

        // Handle tapping a .kcp or .3p file from Android file manager
        intent?.data?.let { onFileSelected(it) }
    }

    private fun onFileSelected(uri: Uri) {
        if (mode == Mode.BUSY) return
        val (name, size) = queryMeta(uri)
        selectedUri = uri
        selectedName = name
        selectedSize = size
        val lower = name.lowercase()
        isExtract = lower.endsWith(".kcp") || lower.endsWith(".3p")

        tvSelectedLabel.text = if (isExtract) "Selected .kcp archive:" else "Selected file:"
        tvFileName.text = name
        tvBefore.text = fmtSize(size)
        tvAfter.text = "0 B"
        tvSaved.text = "0%"
        progressBar.progress = 0
        tvStatus.text = if (isExtract) {
            "Ready to extract and remove the .kcp archive."
        } else {
            "Ready to convert to .kcp and remove the original file."
        }
        mode = Mode.READY
        btnAction.text = if (isExtract) "Extract .kcp" else "Convert to .kcp"
    }

    private fun startConversion() {
        val uri = selectedUri ?: return
        mode = Mode.BUSY
        btnAction.text = "Cancel"
        progressBar.progress = 0
        tvStatus.text = if (isExtract) "Extracting..." else "Compressing to .kcp..."

        thread(name = "3p-worker") {
            val workDir = File(cacheDir, "job_${System.currentTimeMillis()}").apply { mkdirs() }
            val inFile = File(workDir, selectedName.ifBlank { "input.bin" })
            val outDir = File(workDir, "out").apply { mkdirs() }

            try {
                contentResolver.openInputStream(uri)?.use { inp ->
                    FileOutputStream(inFile).use { out -> inp.copyTo(out) }
                } ?: throw IllegalStateException("Cannot read selected file.")

                val resStr = NativeEngine.processFile(
                    inFile.absolutePath,
                    outDir.absolutePath,
                    isExtract,
                    object : ProgressCallback {
                        override fun onProgress(fraction: Double, outBytes: Long) {
                            runOnUiThread {
                                progressBar.progress = (fraction * 100.0).toInt().coerceIn(0, 100)
                                tvAfter.text = fmtSize(outBytes)
                            }
                        }
                    }
                )

                val parts = resStr.split("|")
                when (parts.firstOrNull()) {
                    "OK" -> {
                        val outFile = File(parts[1])
                        val inSz = parts[2].toLongOrNull() ?: selectedSize
                        val outSz = parts[3].toLongOrNull() ?: outFile.length()

                        // Save converted .kcp (or restored file) to Downloads
                        saveToDownloads(outFile)

                        // Remove the original file only AFTER converted file is safely saved
                        val deleted = try {
                            DocumentsContract.deleteDocument(contentResolver, uri)
                        } catch (_: Exception) {
                            false
                        }

                        runOnUiThread {
                            progressBar.progress = 100
                            tvBefore.text = fmtSize(inSz)
                            tvAfter.text = fmtSize(outSz)
                            val pct = if (!isExtract && inSz > 0) {
                                ((1.0 - outSz.toDouble() / inSz.toDouble()) * 100.0).coerceIn(0.0, 100.0)
                            } else 0.0
                            tvSaved.text = "${pct.toInt()}%"
                            tvStatus.text = if (deleted) {
                                "Saved ${outFile.name} to Downloads and removed original."
                            } else {
                                "Saved ${outFile.name} to Downloads (original could not be deleted by provider)."
                            }
                            mode = Mode.HOME
                            btnAction.text = "Browse"
                        }
                    }
                    "CANCELLED" -> runOnUiThread {
                        tvStatus.text = "Cancelled."
                        mode = Mode.READY
                        btnAction.text = if (isExtract) "Extract .kcp" else "Convert to .kcp"
                    }
                    else -> {
                        val msg = parts.getOrNull(1) ?: "Conversion failed."
                        runOnUiThread {
                            tvStatus.text = "Error: $msg"
                            mode = Mode.HOME
                            btnAction.text = "Browse"
                        }
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    tvStatus.text = "Error: ${e.message}"
                    mode = Mode.HOME
                    btnAction.text = "Browse"
                }
            } finally {
                workDir.deleteRecursively()
            }
        }
    }

    private fun saveToDownloads(file: File) {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, file.name)
            put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream")
            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val dstUri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("Cannot create output file in Downloads.")
        contentResolver.openOutputStream(dstUri)?.use { out ->
            FileInputStream(file).use { inp -> inp.copyTo(out) }
        }
    }

    private fun queryMeta(uri: Uri): Pair<String, Long> {
        var name = "file"
        var size = 0L
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                val ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val si = c.getColumnIndex(OpenableColumns.SIZE)
                if (ni >= 0) name = c.getString(ni) ?: name
                if (si >= 0) size = c.getLong(si)
            }
        }
        return name to size
    }

    private fun fmtSize(bytes: Long): String {
        if (bytes <= 0L) return "0 B"
        val units = arrayOf("B", "KB", "MB", "GB")
        var v = bytes.toDouble()
        var u = 0
        while (v >= 1024.0 && u < units.lastIndex) { v /= 1024.0; u++ }
        return if (u == 0) "$bytes B" else String.format("%.1f %s", v, units[u])
    }
}
