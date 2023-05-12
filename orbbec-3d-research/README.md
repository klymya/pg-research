# Orbbec 3D research

We store here the results of [Orbbec Persee 3D camera](https://orbbec3d.com/index/Product/info.html?cate=38&id=37) research and brief descriptions for it.

## Camera setup
### OS instalation
The process of instalation of Ubuntu on the device describes in [the wiki](https://dev.azure.com/pointgrab/PG/_wiki/wikis/PG.wiki/341/Orbbec-Persee).

This distributive already has OpenNI2, GL and LibUSB libs, the libraries headers are here:
```
/home/ubuntu/orbbecAPI_rev1.0/orbbecintegratesimple_rev1.0/ThirdParty
```

### OpenCV instalation
Also you need to install OpenCV for more comfortable work. We used version 4.5.5.

Installation instruction from OpenCV docs: https://docs.opencv.org/4.x/d4/d65/tutorial_orbbec_astra.html
The `OpenNIDevEnvironment` file from the instruction is in this repository.

## Code examples
The distributive has code examples also. The examples placed here:
```
/home/ubuntu/orbbecAPI_rev1.0/orbbecintegratesimple_rev1.0/src
```
And already built app:
```
/home/ubuntu/orbbecAPI_rev1.0/orbbecintegratesimple_rev1.0/Platform/Linux(ARM)/orbbecinteragesimple_rev1.0
```

## Artifacts

### Depth map

The 1st step was to investigate the capabilities of OpenNI2 and OpenCV API. The target was to grab video streams from RGB and depth cameras simultaneously.

The OpenNI2 can read only depth and IR camera stream and OpenCV built with OpenNI2 support can read both RGB and depth sources.

The source code placed:
```
src/deepth
```
The example also required variables defined in `OpenNIDevEnvironment`.

The CMake code for searching OpenNI based on the source:
https://searchcode.com/file/328611440/GUI/FindOpenNI2.cmake/

#### Depth map using OpenNI2 and OpenCV
The example for reading depth source using OpenNI2 and RGB using OpenCV based on:
https://stackoverflow.com/questions/15966722/capture-rgb-from-kinnect-with-openni-and-show-with-opencv

The code here:
```
src/depth/openni
```

#### Depth map using OpenCV
The example for reading both sources using OpenCV based on:
https://opencv.org/working-with-orbbec-astra-3d-cameras-using-opencv/

The code here:
```
src/depth/opencv
```
