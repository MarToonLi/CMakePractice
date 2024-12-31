
find_package(Qt5 COMPONENTS Core REQUIRED)
add_library(OCR_Algo SHARED "")

target_sources(OCR_Algo
PUBLIC 
    foundation.h
    implfoundation.h
    typec_ocr.h
    typec_ocr_prepostprocess.h
    ransac2d.h
    yolov5.hpp
    icroland-scratch-grade.h
    CvxText.h
PRIVATE 
    foundation.cpp
    implfoundation.cpp
    typec_ocr_run.cpp
    ransac2d.cpp
    yolov5.cpp 
    icroland-scratch-grade.cpp
    CvxText.cpp    # if PUBLIC, ng why
)

target_include_directories(OCR_Algo
PUBLIC
    ./
PRIVATE
    ./
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/include/onnxruntime/core/session
    ${THIRDPARTY_DIR}/freetype_2_13_1/include
    ${THIRDPARTY_DIR}/zhicheng/include
    ${THIRDPARTY_DIR}/zhicheng/include/halconcpp
    ${THIRDPARTY_DIR}/cuda118/include
)
target_link_directories(OCR_Algo
PUBLIC
    ./
PRIVATE
    ${THIRDPARTY_DIR}/freetype_2_13_1/objs/x64/Release
)

target_link_libraries(OCR_Algo
PUBLIC 
    algosettingdlg
    logging
    common
    base_alg
    Qt5::Core
    yaml-cpp
PRIVATE
    algosettingdlg
    debug typec_ocr_prepostprocess_D.lib
    optimized typec_ocr_prepostprocess_R.lib
    ${THIRDPARTY_DIR}/onnxruntime_1_17_cuda118/build/Windows/Release/Release/onnxruntime.lib
    ${THIRDPARTY_DIR}/freetype_2_13_1/objs/x64/Release/freetype.lib
    ${THIRDPARTY_DIR}/zhicheng/lib/x64-win64/halconcpp.lib
    ${THIRDPARTY_DIR}/cuda118/lib//x64//nvml.lib

)