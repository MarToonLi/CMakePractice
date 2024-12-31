include(FetchContent)

if(THIRDPARTY_DIR)
FetchContent_Declare(spdlog                                 # FetchContent_Declare����Ҫ���ص�����
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
# FetchContent_MakeAvailable���ز�����spdlog
# �����彫spdlog�⹹���ɺ���Ŀ�꣬ȡ����spdlog��cmakelist�ļ� 
# spdlog��cmakelist�ļ� Ĭ�ϻ����� spdlog ��̬��