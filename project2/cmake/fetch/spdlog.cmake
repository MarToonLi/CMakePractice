include(FetchContent)

if(THIRDPARTY_DIR)
FetchContent_Declare(spdlog                                 # FetchContent_Declare声明要下载的内容
    GIT_REPOSITORY ${THIRDPARTY_DIR}/spdlog
    GIT_TAG v1.11.0
)
else()
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.11.0
)
endif()

FetchContent_MakeAvailable(spdlog)                         
# FetchContent_MakeAvailable下载并构建spdlog
# 而具体将spdlog库构建成何种目标，取决于spdlog的cmakelist文件 
# spdlog的cmakelist文件 默认会生成 spdlog 静态库