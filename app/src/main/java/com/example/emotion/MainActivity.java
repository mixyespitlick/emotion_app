package com.example.emotion;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.emotion.ml.AngryHappySadModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_CODE = 1;
    private static final int CAPTURE_CODE = 2;
    private static final int OPEN_IMAGE_MEDIA_CODE = 3;

    Button takePicture, openGallery;
    ImageView inputImage;
    TextView prediction;
    Uri image_uri;
    int imageSize = 256;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        takePicture = findViewById(R.id.btnOpenCamera);
        openGallery = findViewById(R.id.btnOpenGallery);
        inputImage = findViewById(R.id.imgVInputImage);
        prediction = findViewById(R.id.txtPrediction);

        takePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    int permissionCheckCamera = ActivityCompat.checkSelfPermission(MainActivity.this,Manifest.permission.CAMERA);
                    int permissionStore = ActivityCompat.checkSelfPermission(MainActivity.this,Manifest.permission.WRITE_EXTERNAL_STORAGE);
                    if (permissionCheckCamera != PackageManager.PERMISSION_GRANTED || permissionStore !=PackageManager.PERMISSION_GRANTED) {
//                       ActivityCompat.requestPermissions(MainActivity.this,new String[]{android.Manifest.permission.CAMERA},CAMERA_PERMISSION_CODE);
                        String[] permission = {Manifest.permission.CAMERA,Manifest.permission.WRITE_EXTERNAL_STORAGE};
                        requestPermissions(permission,PERMISSION_CODE);
                    } else {
//                        openCamera();
                        Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(camIntent,CAPTURE_CODE);
                    }
                }
            }
        });

        openGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent storageIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(storageIntent,OPEN_IMAGE_MEDIA_CODE);
            }
        });


    }

    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE,"new image");
        values.put(MediaStore.Images.Media.DESCRIPTION,"camera");

        image_uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values);

        Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        camIntent.putExtra(MediaStore.EXTRA_OUTPUT,image_uri);
        startActivityForResult(camIntent,CAPTURE_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case PERMISSION_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    openCamera();
                } else {
                    Toast.makeText(this,"Permission denied",Toast.LENGTH_SHORT).show();
                }
        }
    }

    @SuppressLint("MissingSuperCall")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK) {
            inputImage.setImageURI(image_uri);
        }

        if (requestCode == CAPTURE_CODE) {
            Bitmap bitmapImage = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(bitmapImage.getWidth(),bitmapImage.getHeight());
            bitmapImage = ThumbnailUtils.extractThumbnail(bitmapImage,dimension,dimension);
            inputImage.setImageBitmap(bitmapImage);

            bitmapImage = Bitmap.createScaledBitmap(bitmapImage,imageSize,imageSize,false);
            classifyImage(bitmapImage);
        } else {
            Uri uri = data.getData();
            Bitmap bitmapImage = null;

            try {
                bitmapImage = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
            inputImage.setImageBitmap(bitmapImage);
            bitmapImage = Bitmap.createScaledBitmap(bitmapImage,imageSize,imageSize,false);
            classifyImage(bitmapImage);
        }
    }

    private void classifyImage(Bitmap bitmapImage) {
        try {
            AngryHappySadModel model = AngryHappySadModel.newInstance(MainActivity.this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);
            // inputFeature0.loadBuffer(byteBuffer);

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmapImage);
            ByteBuffer byteBuffer = tensorImage.getBuffer();

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            AngryHappySadModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0;i<confidences.length;i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Angry","Happy","Sad"};
            prediction.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}