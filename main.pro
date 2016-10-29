TEMPLATE += app
RESOURCES = main.qrc
QT += gui declarative

CONFIG += link_pkgconfig
PKGCONFIG += opencv

L += -L/usr/lib \
-lopencv_core \
-lopencv_imgproc \
-lopencv_highgui \
-lopencv_ml \
-lopencv_video \
-lopencv_features2d \
-lopencv_calib3d \
-lopencv_objdetect \
-lopencv_contrib \
-lopencv_legacy \
-lopencv_flann \
-lcudart\
-L/usr/local/cuda-5.5/lib \

INCLUDEPATH += /usr/include/opencv2 \
INCLUDEPATH += /usr/local/cuda-5.5/include \
INCLUDEPATH += .

CUDA_SOURCES += blur.cu
CUDA_SOURCES += bw.cu
CUDA_SOURCES += SobelFilter.cu

CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')

INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib
LIBS += -lcudart

cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o#bj

cuda.commands = nvcc -c   $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#cuda.depends = nvcc -M  $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," |\ sed "s,^ #*,," | tr -d '\n'



HEADERS       = imageviewer.h 	blur.h bw.h SobelFilter.h

SOURCES       = imageviewer.cpp  main.cpp 


cuda.input = CUDA_SOURCES

QMAKE_EXTRA_COMPILERS += cuda
