# Udacity's Intro to parallel programming course

## Installation Steps
+ Install opencv, no need to bother with PATH variables, just hard code it to the cmakelists file
+ Just remember to set your [path env variable correctly](https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html#windowssetpathandenviromentvariable), otherwise you'll have to sett OPENCV_DIR mannually D:
+ Install VS BuildTools, tested with 15.4
+ Install cudaToolkit, 9.0 coz Tensorflow likes it.
+ Install cmake and add to PATH

## To be able to build cuda in win10:
Use cmake and then msbuild, do not forget to set up PATH properly
```bash
cd "Problem Set {n}"
mkdir build && cd build
cmake ..
msbuild .\HW{n}.sln
```

Remember `msbuild /t:Build /p:Configuration=Release .\HW{n}.sln` for Release :D

### Enjoy
